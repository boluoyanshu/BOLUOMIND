import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim:int,end:int,rope_base:float=1e6,rope_scaling:dict=None):
    freqs,atten_factor=1.0 / (rope_base ** (torch.arange(0,dim,2)[:dim//2].float()/dim)),1.0
    if rope_scaling is not None:
        orig_max,factor,beta_slow,beta_fast,atten_factor=(rope_scaling.get("original_max_position_embeddings",2048),rope_scaling.get("factor",16.0),
                                                          rope_scaling.get("beta_slow",1.0),rope_scaling.get("beta_fast",32.0),rope_scaling.get("attention_factor",1.0))
        if end > orig_max:
            inv=lambda b: (dim * math.log(orig_max / 2 * math.pi * b ))/(2 * math.log(rope_base))
            high = min(math.ceil(inv(beta_slow)),dim//2-1)
            low = max(math.floor(inv(beta_fast)),0)
            ramp=torch.clamp(torch.arange(dim//2).float()-low/(max(high-low,0.01)),0,1)
            freqs=freqs*(1-ramp+ramp/factor)
    t= torch.arange(end,device=freqs.device)
    angles=torch.outer(t,freqs).float()
    freqs_cos=torch.cat([torch.cos(angles),torch.cos(angles)],dim=-1)*atten_factor
    freqs_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)*atten_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        x1= x[...,:x.shape[-1]//2]
        x2= x[...,x.shape[-1]//2:]
        return torch.cat([-x2,x1],dim=-1)
    q_embed= q *cos.unsqueeze(unsqueeze_dim)+rotate_half(q)*sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    return q_embed,k_embed

q=torch.randn(2,4,2,8)
k=torch.randn(2,4,2,8)
cos,sin=precompute_freqs_cis(8,4)
q_embed,k_embed=apply_rotary_pos_emb(q,k,cos,sin)
print(q_embed.size())
rope_scaling = {}
cos, sin = precompute_freqs_cis(8, 4, rope_scaling=rope_scaling)
q_embed,k_embed=apply_rotary_pos_emb(q,k,cos,sin)
print(q_embed.size())