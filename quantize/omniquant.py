import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear, prune_wrapper
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
import numpy as np
import heapq
from parallel_utils import nvidia_smi_memory_info
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state,\
                            prune_parameters, register_prune_masks, clamp_prune_masks, check_val_is_not_inf_nan, enable_debug, \
                            grad_magnitude, log_grads, variational_parameters, eval_variational_reg
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")

def get_memory_usage(logger):
    memory_info = nvidia_smi_memory_info()[0]
    t = memory_info['total_memory']
    a = memory_info['used_memory']
    f = memory_info['free_memory']
    logger.info(f"=== Memory usage ===")
    logger.info(f"Total memory: {t} MB")
    logger.info(f"Allocated memory: {a} MB")
    logger.info(f"Free memory: {f} MB")

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}

def get_magnitudes():
    return prune_wrapper.get_magnitudes()

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     
import torch

@torch.no_grad()
def _sqsum(x):  # utility
    return (x.float() ** 2).sum().item()

def estimate_layer_sensitivity_efisher(model, layer, dataloader, steps=8, device="cuda"):
    """
    Empirical Fisher trace proxy: average squared gradient norm of the layer's params
    over a few small language-modeling mini-batches (teacher forcing).
    """
    was_training = model.training
    model.train()  # enable grads to accumulate EFisher
    total, n = 0.0, 0
    params = [p for p in layer.parameters() if p.requires_grad]
    if not params:
        return 0.0

    it = iter(dataloader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            break

        model.zero_grad(set_to_none=True)
        # Your dataloader yields tokens at batch[0]; use labels = input_ids for LM loss.
        input_ids = batch[0].to(device)
        attention_mask = None
        # Try to build a dict HF models accept; attention_mask is optional here.
        inputs = {"input_ids": input_ids, "labels": input_ids}
        outputs = model(**inputs)   # HF models return .loss when labels present
        loss = outputs.loss
        loss.backward()

        s = 0.0
        for p in params:
            if p.grad is not None:
                s += _sqsum(p.grad)
        total += s
        n += 1

    model.train(was_training)
    return total / max(n, 1)


def allocate_epochs_by_sensitivity(scores, total_budget, min_ep=1, max_ep=8,
                                   mode="pow", gamma=2.0, temperature=0.7):
    """
    Convert nonnegative sensitivity scores to integer epoch budgets.
    mode:
      - "pow":      w_i = s_i ** gamma
      - "softmax":  w_i = softmax( log(s_i) / temperature )
    gamma>1 or temperature<1 sharpen differences.
    """
    eps = 1e-12
    s = [max(float(x), eps) for x in scores]

    if mode == "pow":
        w = [si ** gamma for si in s]
    elif mode == "softmax":
        z = [math.log(si) for si in s]
        z = [zi / max(temperature, 1e-6) for zi in z]
        m = max(z)
        e = [math.exp(zi - m) for zi in z]
        S = sum(e) + eps
        w = [ei / S for ei in e]
    else:
        w = s  # fallback: proportional

    W = sum(w) + eps
    alloc = [min(max_ep, max(min_ep, round(total_budget * wi / W))) for wi in w]

    # Fix rounding drift to match total_budget
    diff = sum(alloc) - total_budget
    if diff > 0:
        # trim from least-weighted
        order = sorted(range(len(alloc)), key=lambda i: w[i])
        for i in order:
            if diff == 0: break
            if alloc[i] > min_ep:
                alloc[i] -= 1; diff -= 1
    elif diff < 0:
        # add to most-weighted
        order = sorted(range(len(alloc)), key=lambda i: -w[i])
        for i in order:
            if diff == 0: break
            if alloc[i] < max_ep:
                alloc[i] += 1; diff += 1
    return alloc

# ==== Delta-loss sensitivity helpers =========================================
@torch.no_grad()
def _collect_eval_batches(dataloader, steps, device):
    """Grab a small list of input_ids tensors for scoring; we will also consume
    these 'steps' from the dataloader (that's OK; nsamples is large)."""
    batches = []
    it = iter(dataloader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            break
        batches.append(b[0].to(device))  # your dataloader's first item is input_ids
    return batches

@torch.no_grad()
def _avg_lm_loss(model, input_list):
    """Compute mean LM loss on a small list of input_ids, using teacher forcing."""
    losses = []
    for input_ids in input_list:
        out = model(input_ids=input_ids, labels=input_ids)
        losses.append(float(out.loss))
    return sum(losses) / max(len(losses), 1)

def _layer_delta_loss_once(model, layers_ref, layer_idx, DecoderLayer, args, device, is_llama, eval_inputs):
    """Quantize ONLY layer_idx, measure Δloss = L_q - L_fp on eval_inputs, then restore."""
    # 1) Build quantized wrapper from the FP layer copy
    fp_layer = layers_ref[layer_idx]
    qlayer = DecoderLayer(model.config, fp_layer, args).to(device)
    # turn on quant for both weights & activations and actually pack scales/zeros
    set_quant_state(qlayer, weight_quant=True, act_quant=True)
    qlayer.half()
    smooth_and_quant_inplace(qlayer, args, is_llama)
    # 2) Swap into model
    layers_ref[layer_idx] = qlayer
    try:
        L_q = _avg_lm_loss(model, eval_inputs)
    finally:
        # 3) Restore original FP layer regardless of errors
        layers_ref[layer_idx] = fp_layer
    return L_q
# ============================================================================

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False

    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    # ---------- Layer sensitivity via Δloss (quantize one layer at a time) ---
    per_layer_alloc = None
    if getattr(args, "use_hessian_alloc", False):
        logger.info("Scoring per-layer sensitivity with Δloss (quantize-only-that-layer)...")
        # Move the whole model to GPU so we can run LM loss cheaply
        model.to(dev)
        # Collect a few small eval batches; this consumes 'trace_batches' from the dataloader.
        trace_steps = max(2, getattr(args, "trace_batches", 8))
        eval_inputs = _collect_eval_batches(dataloader, steps=trace_steps, device=dev)

        # Baseline FP loss (shared for all layers) on the same inputs
        model.eval()
        with torch.no_grad():
            L_fp = _avg_lm_loss(model, eval_inputs)
        logger.info(f"[sens] Baseline FP loss on {len(eval_inputs)} batches: {L_fp:.6f}")

        sens_scores = []
        for li in range(len(layers)):
            # Quantize only layer li, measure L_q, then Δ = max(L_q - L_fp, 0)
            L_q = _layer_delta_loss_once(model, layers, li, DecoderLayer, args, dev, is_llama, eval_inputs)
            delta = max(L_q - L_fp, 0.0)
            sens_scores.append(delta)
            logger.info(f"[sens] layer {li}: L_q={L_q:.6f}, Δ={delta:.6f}")

        # Decide the total epoch budget
        total_budget = getattr(args, "budget_epochs", 0) or (len(layers) * max(1, args.epochs))

        # Use the appropriate allocator based on mode
        if getattr(args, "alloc_mode", "pow") == "powdr":
            per_layer_alloc = alloc_pow_dr(
                sens=sens_scores,
                budget_epochs=total_budget,
                min_layer_epochs=getattr(args, "min_layer_epochs", 1),
                gamma=getattr(args, "alloc_gamma", 2.0),
                temperature=getattr(args, "alloc_temperature", 0.7),
                beta=getattr(args, "alloc_beta", 0.95)
            )
        else:
            # Use your existing allocator for pow/softmax modes
            per_layer_alloc = allocate_epochs_by_sensitivity(
                sens_scores,
                total_budget,
                min_ep=getattr(args, "min_layer_epochs", 1),
                max_ep=getattr(args, "layer_max_epochs", max(1, args.epochs)),
                mode=getattr(args, "alloc_mode", "pow"),
                gamma=getattr(args, "alloc_gamma", 2.0),
                temperature=getattr(args, "alloc_temperature", 0.7),
            )
        logger.info(f"[sens] min={min(sens_scores):.6f} max={max(sens_scores):.6f} "
                    f"ratio={(max(sens_scores)/(min(sens_scores)+1e-12)):.2f}")
        logger.info(f"[alloc] all per-layer epochs: {per_layer_alloc}")
    # -------------------------------------------------------------------------
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)

        if args.prune:
            prune_wrapper.set_new_prune_params(prune_method=args.prune_method, prune_active=False, resume=args.resume, n_gumbel_samples=args.ngumbel_samples, n_epochs=args.epochs, dev=dev)
            if args.prune_method == 'pman':
                prune_wrapper.init_new_pman_state()

        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)



        # Pruning

        if args.prune:
            prune_wrapper.set_new_prune_params(prune_method=args.prune_method, prune_active=False, resume=args.resume, n_gumbel_samples=args.ngumbel_samples, n_epochs=args.epochs, dev=dev)
            if args.prune_method == 'pman':
                prune_wrapper.init_new_pman_state()
            else:
                prune_wrapper.set_variational_train_clip(args.var_train_clip)

        # End pruning

        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        # init smooth parameters

        # Pruning

        if args.prune:
            prune_wrapper.set_new_prune_params(prune_method=args.prune_method, prune_active=args.prune, resume=args.resume, n_gumbel_samples=args.ngumbel_samples, n_epochs=args.epochs, dev=dev)
            if args.prune_method == 'pman':
                prune_wrapper.init_new_pman_state()

        # End pruning

        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            params = [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr},
                        {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}]
            if args.prune and args.prune_method == 'variational':
                params.append({'params':variational_parameters(qlayer), 'lr':args.var_lr})
            optimizer = torch.optim.AdamW(params,weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()

            # Pruning:

            if args.prune:
                prune_optimizer = None
                if args.prune_method == 'pman':
                    prune_optimizer = torch.optim.SGD([{"params":prune_parameters(qlayer),"lr":args.lwc_lr}])
                    prune_wrapper.init_new_pman_state()
                #if args.prune_method == 'variational':
                #    prune_optimizer = torch.optim.AdamW([{'params':variational_parameters(qlayer), 'lr':args.var_lr}])
                        # --- Adaptive per-layer epochs (early stopping) ---
            use_auto = getattr(args, "auto_layer_epochs", False)

            # If we computed HAWQ-style allocation, use it as the cap for this layer
            if 'per_layer_alloc' in locals() and per_layer_alloc is not None:
                target_epochs = per_layer_alloc[i]
            else:
                # fallback: your previous behavior
                target_epochs = (args.layer_max_epochs if use_auto else args.epochs)
                if use_auto and getattr(args, "budget_epochs", 0) > 0:
                    target_epochs = min(target_epochs, args.layer_max_epochs)

            # never below the declared minimum
            target_epochs = max(target_epochs, getattr(args, "min_layer_epochs", 1))

            prev_loss_mean = float("inf")
            no_improve = 0
            used_epochs = 0

            while used_epochs < target_epochs:
                loss_list = []
                norm_list = []

                # --- Pruning (per-epoch hooks) ---
                if args.prune and args.prune_method == 'pman':
                    prune_optimizer.zero_grad()
                    prune_wrapper.get_prune_pman_state().next_epoch()

                pman_batch_size = args.gumbel_batch_size
                # --- End pruning ---

                # === One epoch over calibration ===
                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)

                        quant_out = qlayer(
                            quant_inps[index:index+args.batch_size,],
                            attention_mask=attention_mask_batch,
                            position_ids=position_ids
                        )[0]

                        reg_loss = 0
                        if args.prune and args.prune_method == 'variational':
                            reg_loss = eval_variational_reg(qlayer)

                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)

                        # --- Pruning inner steps (unchanged) ---
                        prune_loss = 0
                        if args.prune and args.prune_method == 'pman':
                            prune_wrapper.set_train_prune(train_prune=True)
                            for s in range(args.ngumbel_samples):
                                prune_quant_out = qlayer(
                                    quant_inps[index:index+args.batch_size,],
                                    attention_mask=attention_mask_batch,
                                    position_ids=position_ids
                                )[0]
                                prune_loss += loss_func(fp_inps[index:index+args.batch_size,], prune_quant_out) / args.ngumbel_samples
                                prune_wrapper.get_prune_pman_state().next_sample()

                                if (s + 1) % pman_batch_size == 0:
                                    prune_loss.backward(retain_graph=True)
                                    logger.info(f'Grad maximum value for layer {i} is: {grad_magnitude(qlayer, i)}')
                                    prune_optimizer.step()
                                    prune_optimizer.zero_grad()
                                    prune_loss = prune_loss.detach()
                                    prune_loss = 0
                                    clamp_prune_masks(qlayer)

                            prune_wrapper.set_train_prune(train_prune=False)
                            prune_wrapper.get_prune_pman_state().next_batch()

                            should_step = args.ngumbel_samples % pman_batch_size != 0
                            if should_step:
                                prune_loss.backward(retain_graph=True)
                                logger.info(f'Grad maximum value for layer {i} is: {grad_magnitude(qlayer, i)}')
                                prune_optimizer.step()
                                prune_optimizer.zero_grad()
                                clamp_prune_masks(qlayer)
                        # --- End pruning inner steps ---

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()

                    loss_list.append(loss.detach().cpu())
                    loss = loss + reg_loss  # include regularizer in the optimizer step
                    optimizer.zero_grad()
                    param, param_to_name = get_omni_parameters(qlayer, use_shift, args.prune_method == 'variational')
                    norm = loss_scaler(loss, optimizer, parameters=param).cpu()
                    norm_list.append(norm.data)

                # === End epoch: logging + early stopping ===
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} epoch {used_epochs} loss:{loss_mean} norm:{norm_mean} "
                            f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device)/1024**2} ")

                if use_auto:
                    # relative improvement vs previous epoch
                    prev = prev_loss_mean
                    cur = loss_mean.item()
                    gain = 1.0 if not math.isfinite(prev) else (prev - cur) / (abs(prev) + 1e-8)

                    # update trackers
                    if not math.isfinite(prev) or gain >= args.layer_min_gain:
                        no_improve = 0
                        prev_loss_mean = cur
                    else:
                        no_improve += 1

                    if no_improve >= args.layer_patience:
                        logger.info(f"[Layer {i}] Early stop at epoch {used_epochs+1}: "
                                    f"gain<{args.layer_min_gain} for {args.layer_patience} epochs")
                        break
                else:
                    prev_loss_mean = loss_mean.item()

                used_epochs += 1
            # --- End adaptive loop ---

            # End pruning
            """           
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                # Pruning:

                if args.prune:
                    if args.prune_method == 'pman':
                        prune_optimizer.zero_grad()
                        prune_wrapper.get_prune_pman_state().next_epoch()
                    #enable_debug(qlayer, i, logger, epochs, args.epochs)

                pman_batch_size = args.gumbel_batch_size

                # End pruning
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)

                        #if args.prune and args.prune_method == 'variational':
                            #prune_wrapper.set_train_prune(train_prune=True)    
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        reg_loss = 0
                        if args.prune and args.prune_method == 'variational':
                            reg_loss =  eval_variational_reg(qlayer)
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)

                        # Pruning:

                        prune_loss = 0
                        if args.prune:
                            if args.prune_method == 'pman':
                                prune_wrapper.set_train_prune(train_prune=True)

                                for s in range(args.ngumbel_samples):
                                    prune_quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                                    prune_loss += loss_func(fp_inps[index:index+args.batch_size,], prune_quant_out) / args.ngumbel_samples
                                    prune_wrapper.get_prune_pman_state().next_sample()

                                    if (s + 1) % pman_batch_size == 0:
                                        prune_loss.backward(retain_graph=True)
                                        logger.info(f'Grad maximum value for layer {i} is: {grad_magnitude(qlayer, i)}')
                                        prune_optimizer.step()
                                        prune_optimizer.zero_grad()
                                        prune_loss = prune_loss.detach()
                                        prune_loss = 0
                                        #check_val_is_not_inf_nan(qlayer, logger, i, j, s)
                                        clamp_prune_masks(qlayer)

                                prune_wrapper.set_train_prune(train_prune=False)
                                prune_wrapper.get_prune_pman_state().next_batch()

                                should_step = args.ngumbel_samples % pman_batch_size != 0

                                
                                if should_step:
                                    prune_loss.backward(retain_graph=True)
                                    logger.info(f'Grad maximum value for layer {i} is: {grad_magnitude(qlayer, i)}')
                                    prune_optimizer.step()
                                    prune_optimizer.zero_grad()
                                    clamp_prune_masks(qlayer)
                            
                            if args.prune_method == 'variational':
                                prune_loss = eval_variational_reg(qlayer)
                                prune_loss.backward()
                                #logger.info(f'Grad maximum value for layer {i} is: {grad_magnitude(qlayer, i)}')
                                prune_optimizer.step()
                                #logger.info(f'Grad maximum value for layer {i} is: {grad_magnitude(qlayer, i)}')
                                prune_optimizer.zero_grad()
                                #logger.info(f'Grad maximum value for layer {i} is: {grad_magnitude(qlayer, i)}')
                        # End pruning

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    loss += reg_loss
                    optimizer.zero_grad()
                    param, param_to_name = get_omni_parameters(qlayer, use_shift, args.prune_method == 'variational')
                    norm = loss_scaler(loss, optimizer,parameters=param).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
            """
            # Pruning

            if args.prune:
                del prune_optimizer

            # End Pruning

        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            # Pruning:

            if args.prune and args.prune_method == 'variational':
                enable_debug(qlayer, i, logger, args.epochs, args.epochs)
            #    if args.prune_method is 'pman':
            #        register_prune_masks(qlayer)

            # End pruning
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer, use_prune=args.prune)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)

            # Pruning:

            if args.prune and args.prune_method == 'variational':
                enable_debug(qlayer, i, logger, args.epochs, args.epochs)
            #    if args.prune_method is 'pman':
            #        register_prune_masks(qlayer)

            # End pruning
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    if args.prune and args.prune_method == 'pman':
        log_grads(args.output_dir, logger)
    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

def _softmax_logits(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    z = np.exp(logits / max(temperature, 1e-6))
    return z / np.sum(z)

def alloc_pow_dr(sens, budget_epochs, min_layer_epochs, gamma, temperature, beta):
    """
    Power-weighted diminishing-returns allocator (cap-less).
    - sens: per-layer sensitivity (higher => more important)
    - gamma: concentration (like your pow mode)
    - temperature: softens the initial weights
    - beta: diminishing-returns exponent (gain ~ w / k^beta for the k-th extra epoch)
    """
    L = len(sens)
    assert budget_epochs >= L * min_layer_epochs, \
        "budget_epochs must cover the per-layer minimum."

    # Power weighting + temperature smoothing (works like a soft prior)
    base = np.maximum(np.asarray(sens, dtype=np.float64), 1e-12)
    poww = np.power(base, gamma)
    # Use log(poww) as logits, then temperature-softmax to avoid numeric issues
    prior = _softmax_logits(np.log(poww + 1e-12), temperature=temperature)

    # Start from the floor (prevents starvation)
    e = np.full(L, int(min_layer_epochs), dtype=np.int64)
    remaining = int(budget_epochs - np.sum(e))

    # Max-heap over marginal gains; store as (-gain, i, k_next)
    # where k_next = (e_i - min) + 1 for the *next* unit you would add
    heap = []
    for i in range(L):
        k_next = 1  # first extra epoch above the floor
        gain = prior[i] / (k_next ** beta)
        heap.append((-gain, i, k_next))
    heapq.heapify(heap)

    while remaining > 0:
        neg_gain, i, k_next = heapq.heappop(heap)
        e[i] += 1
        remaining -= 1
        k_next += 1
        gain_next = prior[i] / (k_next ** beta)
        heapq.heappush(heap, (-gain_next, i, k_next))

    return e.tolist()

