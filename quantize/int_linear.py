import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer

PRUNE_UPPER_LIMIT = 1
VARIATIONAL_UPPER_LIMIT = 8
VARIATIONAL_LOWER_LIMIT = -8
VARIATIONAL_THRESHOLD = 3

class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        # Pruning:

        self.prune_mask = None
        self.log_sigma2 = None
        #self.prev_param = None
        if prune_wrapper.get_prune_active():
            if prune_wrapper.get_prune_method() == 'pman':
                self.prune_mask = nn.Parameter(torch.clamp(torch.zeros(size=self.weight.shape, dtype=self.weight.dtype, device=prune_wrapper.get_device()).uniform_(), min=0, max=PRUNE_UPPER_LIMIT))

            if prune_wrapper.get_prune_method() == 'variational':
                self.log_sigma2 = nn.Parameter(torch.zeros_like(self.weight, dtype=self.weight.dtype, device=prune_wrapper.get_device()) - 10)
        self.print_debug = False

        # End Pruning:
    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        if prune_wrapper.get_prune_active() and prune_wrapper.get_prune_method() == 'variational':
            if self.log_sigma2 is None:
                    self.log_sigma2 = nn.Parameter(torch.zeros_like(self.weight, dtype=self.weight.dtype, device=prune_wrapper.get_device()) - 10)
            """
            if self.prev_param is None:
                self.prev_param = self.log_sigma2

            else:
                print(f'prev and param are different in {torch.sum((self.prev_param != self.log_sigma2).to(dtype=torch.int))} cells')
                self.prev_param = self.log_sigma2
            """
            log_alpha = torch.clamp((self.log_sigma2 - torch.log(weight * weight)), min=VARIATIONAL_LOWER_LIMIT, max=VARIATIONAL_UPPER_LIMIT)
            clip_mask = torch.lt(log_alpha, VARIATIONAL_THRESHOLD).to(dtype=torch.int)

            if not prune_wrapper.get_train_prune():
                weight = weight * clip_mask

            if self.print_debug:
                self.print_info(prune_wrapper.get_prune_method())
                self.print_debug = False

            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

            #out = self.calculate_variational(weight, input, out, log_alpha, clip_mask)

        else:
            if prune_wrapper.get_prune_active() and prune_wrapper.get_prune_method() == 'pman':
                if self.prune_mask is None:
                    self.prune_mask = nn.Parameter(torch.clamp(torch.zeros(size=self.weight.shape, dtype=self.weight.dtype, device=prune_wrapper.get_device()).uniform_(), min=0, max=PRUNE_UPPER_LIMIT))
                if self.print_debug:
                    self.print_info(prune_wrapper.get_prune_method())
                    self.print_debug = False
                weight = weight * self.calculate_mask()

            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out
    
    def calculate_mask(self):
        mask = None
        if prune_wrapper.get_prune_method() == 'pman':
            if prune_wrapper.get_train_prune():
                tmp_1 = torch.ones(self.prune_mask.shape, dtype=self.prune_mask.dtype, device=prune_wrapper.get_device()) - self.prune_mask
                tmp_2 = torch.log(self.prune_mask / tmp_1)
                mask = torch.sigmoid((tmp_2 + prune_wrapper.get_prune_pman_state().get_gumbel_sample()) / prune_wrapper.get_prune_pman_state().get_temprature())
                mask = mask.to(dtype=self.weight.dtype)
            else:
                mask = torch.bernoulli(self.prune_mask).to(dtype=self.weight.dtype, device=prune_wrapper.get_device())

        return mask

    def calculate_variational(self, weight, input, out, log_alpha, clip_mask):
        res = out

        if prune_wrapper.get_variational_train_clip():
            weight = weight * clip_mask

        mu = out
        tmp1 = input * input
        tmp2 = torch.exp(log_alpha) * weight * weight

        """print(f'input shape: {input.shape}')
        print(f'weight shape: {weight.shape}')
        print(f'out shape: {out.shape}')
        print(f'mu shape: {mu.shape}')
        print(f'tmp1 shape: {tmp1.shape}')
        print(f'tmp2 shape: {tmp2.shape}')
        print(f'log_sigma2 shape: {self.log_sigma2.shape}')"""

        tmp3 = torch.matmul(tmp1, tmp2.T)
        tmp4 = tmp3 + 1e-8
        si = torch.sqrt(tmp4)

        res = mu + (torch.randn_like(mu) * si)

        return res
    
    def get_reg(self):
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695
        C = -k1
        log_alpha = torch.clamp((self.log_sigma2 - torch.log(self.weight * self.weight)), min=VARIATIONAL_LOWER_LIMIT, max=VARIATIONAL_UPPER_LIMIT)
        mdkl = self.eval_reg(log_alpha)
        #mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        #print(f'self = {self}, mdkl = {torch.sum(mdkl)}')
        return torch.sum(mdkl)
    
    def eval_reg(self, log_alpha):
        return (0.5 * torch.log1p(torch.exp(-log_alpha)) - (0.03 + 1.0 / (1.0 + torch.exp(-(1.5 * (log_alpha + 1.3)))) * 0.64))


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def get_sparsity(self, i):
        number_of_weights = 0
        if prune_wrapper.get_prune_method() == 'variational':
            log_alpha = torch.clamp((self.log_sigma2 - torch.log(self.weight * self.weight)), min=VARIATIONAL_LOWER_LIMIT, max=VARIATIONAL_UPPER_LIMIT)
            clip_mask = torch.lt(log_alpha, VARIATIONAL_THRESHOLD).to(dtype=torch.int)
            number_of_weights = torch.sum(clip_mask).item()
            sparsity = (1 - number_of_weights / torch.numel(self.weight)) * 100
            magnitudes = prune_wrapper.get_magnitudes()
            if i not in magnitudes.keys():
                magnitudes[i] = []
            magnitudes[i].append(sparsity)

    def print_info(self, method):
        number_of_weights = 0
        if method == 'pman':
            number_of_weights = torch.sum(self.prune_mask.to(dtype=torch.float32)).item()
            magnitudes = prune_wrapper.get_magnitudes()
            if self.layer_num not in magnitudes.keys():
                magnitudes[self.layer_num] = {}
            magnitudes[self.layer_num][self.debug_str] = self.prune_mask.view(-1).cpu()
        if method == 'variational':
            log_alpha = torch.clamp((self.log_sigma2 - torch.log(self.weight * self.weight)), min=VARIATIONAL_LOWER_LIMIT, max=VARIATIONAL_UPPER_LIMIT)
            clip_mask = torch.lt(log_alpha, VARIATIONAL_THRESHOLD).to(dtype=torch.int)
            number_of_weights = torch.sum(clip_mask).item()
        self.logger.info(self.debug_str)
        self.logger.info(f'Shape: {self.weight.shape}')
        self.logger.info(f'Average number of weights not zero is: {number_of_weights}')
        self.logger.info(f"Percentage of weights on average not zero is: {(number_of_weights / torch.numel(self.weight)) * 100}")

# Pruning:

    def clamp_prune_mask(self):
        with torch.no_grad():
            self.prune_mask.clamp_(min=0, max=PRUNE_UPPER_LIMIT)
    
    def enable_debug(self, str, i, logger):
        self.debug_str = str
        self.print_debug = True
        self.logger = logger
        self.layer_num = i

class PmanState():
    def __init__(self, n_epochs=1, dev="cpu"):
        self.temprature = 1
        if n_epochs == 0:
            n_epochs = 1
        self.temprature_decline = 0.97 / n_epochs
        self.gumbel_samples = torch.zeros(prune_wrapper.get_n_gumbel_samples())
        self.gumbel_index = 0
        self.gumbel_samples = self.gumbel_samples.to(device=dev)
        dist_1 = torch.distributions.gumbel.Gumbel(0, 1)
        dist_2 = torch.distributions.gumbel.Gumbel(0, 1)

        for i in range(self.gumbel_samples.shape[0]):
            self.gumbel_samples[i] = dist_1.sample() - dist_2.sample()

    
    def get_gumbel_sample(self):
        return self.gumbel_samples[self.gumbel_index]
    
    def get_temprature(self):
        return self.temprature
    
    def next_epoch(self):
        self.temprature -= self.temprature_decline
    
    def next_batch(self):
        self.gumbel_index = 0
    
    def next_sample(self):
        self.gumbel_index += 1


class PruneStateWrapper():
    def __init__(self, prune_method='pman', prune_active=False, resume=False, n_gumbel_samples=1024, n_epochs=1, dev="cpu", variational_train_clip=False):
        self.prune_active = prune_active
        self.resume = resume
        self.n_gumbel_samples = n_gumbel_samples
        self.n_epochs = n_epochs
        self.dev = dev
        self.pman_state_obj = None
        self.prune_method = prune_method
        self.magnitudes = {}
        self.variational_train_clip = variational_train_clip
        self.train_prune = False

    def init_new_pman_state(self):
        self.pman_state_obj = PmanState(self.n_epochs, self.dev)

    def set_new_prune_params(self, prune_method='pman', prune_active=False, resume=False, n_gumbel_samples=1024, n_epochs=1, dev="cpu"):
        self.prune_method = prune_method
        self.prune_active = prune_active
        self.resume = resume
        self.n_gumbel_samples = n_gumbel_samples
        self.n_epochs = n_epochs
        self.dev = dev

    def get_prune_pman_state(self):
        return self.pman_state_obj
    
    def get_prune_active(self):
        return self.prune_active
    
    def get_device(self):
        return self.dev
    
    def get_resume(self):
        return self.resume
    
    def get_n_gumbel_samples(self):
        return self.n_gumbel_samples
    
    def get_prune_method(self):
        return self.prune_method
    
    def get_magnitudes(self):
        return self.magnitudes
    
    def set_variational_train_clip(self, variational_train_clip=False):
        self.variational_train_clip = variational_train_clip

    def get_variational_train_clip(self):
        return self.variational_train_clip
    
    def set_train_prune(self, train_prune=False):
        self.train_prune = train_prune

    def get_train_prune(self):
        return self.train_prune


prune_wrapper = PruneStateWrapper()

# End Pruning:
