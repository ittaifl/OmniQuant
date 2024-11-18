from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from models.transformation import *

def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)

# Pruning:

def prune_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('prune') > -1:
            params.append(m)
    return iter(params)

def variational_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('sigma') > -1:
            params.append(m)
    return iter(params)

def eval_variational_reg(model):
    reg = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            reg += module.get_reg()

    return reg


def register_prune_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.register_prune_mask()

def clamp_prune_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.clamp_prune_mask()

def check_val_is_not_inf_nan(model, logger, i, j, s):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            for p_name, param in module.named_parameters():
                if p_name.find('prune') > -1 and (torch.isnan(param).any() or torch.isinf(param).any()):
                    for k in range(param.shape[0]):
                        for r in range(param.shape[1]):
                            if torch.isnan(param[k, r]):
                                logger.info(f"NaN value at prune mask in layer {i}, module {name}, batch {j}, sample {s}")
                                logger.info(f"param name is {p_name}, at index ({k}, {r})")
                            if torch.isinf(param[k, r]):
                                logger.info(f"inf value at prune mask in layer {i}, module {name}, batch {j}, sample {s}")
                                logger.info(f"param name is {p_name}, at index ({k}, {r})")

def enable_debug(model, i, logger, epoch=0, n_epoch=1):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            """str = f'epoch_{epoch}'
            if epoch == n_epoch:
                str = f'inference'
            module.enable_debug(f'layer_{i}_{name}_{str}', i, logger)
            """
            module.get_sparsity(i)


grads = {}
grad_by_sample = {}

def grad_magnitude(model, i):
    max = None
    GRAD_THRESHOLD = 1e-4
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            for p_name, param in module.named_parameters():
                if p_name.find('prune') > -1 or p_name.find('sigma') > -1 and param.grad is not None:
                    #if torch.logical_not(torch.isfinite(param.grad)).any():
                    #    print(f"problem in layer {i}")
                    if i not in grads.keys():
                        grads[i] = {}
                    if name not in grads[i].keys():
                        grads[i][name] = torch.zeros(size = param.view(-1).shape, device='cpu', dtype=param.grad.dtype)
                    d = param.grad.view(-1).cpu()
                    grads[i][name] += d
                    ge_elements = torch.sum(torch.ge(torch.abs(param.grad), GRAD_THRESHOLD).to(dtype=torch.int)).cpu()
                    sparsity_rate = (1 - (ge_elements / torch.numel(param.grad))) * 100
                    if i not in grad_by_sample.keys():
                        grad_by_sample[i] = {}
                    if name not in grad_by_sample[i].keys():
                        grad_by_sample[i][name] = []
                    grad_by_sample[i][name].append(sparsity_rate)
                    #print(p_name)
                    tmp_max = torch.max(torch.abs(param.grad))
                    #print(tmp_max)
                    #print(f'\nname = {p_name}:\n')
                    #print(param.grad)
                    if max is None:
                        max = tmp_max
                    else:
                        if max < tmp_max:
                            tmp_max = max
    
    return max

def log_grads(output_dir, logger):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    sparsity_rates = []
    GRAD_THRESHOLD = 1e-2
    threasholds = [1e-2, 2e-2, 5e-2]
    spartsity_per_layer = {}

    logger.info(f'Calculating sparsity of gradients per layer')
    for t in threasholds:
        spartsity_per_layer[t] = []
        for layer_num, nest_dict in grads.items():
            num_of_weights = 0
            num_of_grads = 0
            # Creating subplots with multiple histograms
            for k, d in nest_dict.items():
                num_of_grads += torch.sum(torch.ge(torch.abs(d), t).to(dtype=torch.int))
                num_of_weights += torch.numel(d)

            spartsity_per_layer[t].append((1 - (num_of_grads / num_of_weights)) * 100)

    for layer_num, nest_dict in grads.items():
        logger.info(f'Creating plots for layer {layer_num}')
        # Creating subplots with multiple histograms
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(48, 8))
        i = 0

        for k, d in nest_dict.items():
            axes[int(i / 2), i % 2].hist(d.numpy(), bins=100, color='Yellow', edgecolor='black')
            axes[int(i / 2), i % 2].set_title(f'Histogram - {k}')
            axes[int(i / 2), i % 2].set_xlabel('Values')
            axes[int(i / 2), i % 2].set_ylabel('Frequency')
            i += 1
            ge_elements = torch.sum(torch.ge(torch.abs(d), GRAD_THRESHOLD).to(dtype=torch.int))
            sparsity_rate = (1 - (ge_elements / torch.numel(d))) * 100
            logger.info(f'Sparsity rate of accumalted gradient for {k} is: {sparsity_rate}')
            sparsity_rates.append(sparsity_rate)
        
        # Adjusting layout for better spacing
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"grads_plot_layer_{layer_num}"))

        plt.clf()

    

    logger.info(f'Total number of linear layers: {len(sparsity_rates)}')
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes.hist(sparsity_rates, bins=100, color='Red', edgecolor='black')
    axes.set_title(f'Grads sparsity rates across all layers')
    axes.set_xlabel('Values')
    axes.set_ylabel('Frequency')

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"grads_sparsity"))

    plt.clf()

    logger.info(f'Total number of linear layers: {len(spartsity_per_layer)}')
    for t in threasholds:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        axes.plot(spartsity_per_layer[t])
        axes.set_title(f'Sparsity of gradients in each layer (threashold={t})')
        axes.set_xlabel('Layer')
        axes.set_ylabel('Grads sparsity')

        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"grads_sparsity_per_layer_threashold_{t}.png"))

        plt.clf()

    for layer_num, nest_dict in grad_by_sample.items():
        logger.info(f'Creating plots for layer {layer_num}')
        # Creating subplots with multiple histograms
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(48, 8))
        i = 0

        for k, d in nest_dict.items():
            axes[int(i / 2), i % 2].scatter(np.arange(len(d)), d)
            axes[int(i / 2), i % 2].set_title(f'Sparsity rate while training - {k}')
            axes[int(i / 2), i % 2].set_xlabel('Number of step')
            axes[int(i / 2), i % 2].set_ylabel('Sparsity rate')
            i += 1
        
        # Adjusting layout for better spacing
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"sparsity_rate_plot_per_sample_layer_{layer_num}"))

        plt.clf()
# End pruning

def get_omni_parameters(model, use_shift=True, use_prune=False):
    params = []
    params_names = []
    template = "smooth" if use_shift else "smooth_scale"
    template2 = 'sigma' if use_prune else "!"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1 or n.find(template2):
            params.append(m)
            params_names.append(n)
    return iter(params), iter(params_names)  

def omni_state_dict(model, destination=None, prefix='', keep_vars=False, use_prune=False):
    if destination is None:
        destination = OrderedDict()
    template = "prune_mask" if use_prune else "!"
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1 or name.find(template):
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, isllama):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                            model.qkt_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
