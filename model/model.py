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

def repeat_kv(x:torch.Tensor,num_rep:int)->torch.Tensor:
    if num_rep==1:
        return x
    return x.repeat_interleave(num_rep,dim=2)
class Attention(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.num_key_value_heads=config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads=config.num_attention_heads
        self.n_local_kv_heads=config.num_key_value_heads
        assert not config.hidden_size%config.num_attention_heads
        self.head_dim=config.head_dim

        self.is_casual=True

        self.q_proj=nn.Linear(config.hidden_size,self.n_local_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(config.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(config.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.out_proj=nn.Linear(self.n_local_heads*self.head_dim,config.hidden_size,bias=False)

        self.q_norm=RMSNorm(config.head_dim,eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        self.atten_dropout=nn.Dropout(config.dropout)
        self.resid_dropout=nn.Dropout(config.dropout)
        self.dropout=config.dropout

        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and config.flash_attn



    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz,seq,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        xq=xq.view(bsz,seq,self.n_local_heads,self.head_dim)
        xk = xk.view(bsz, seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq, self.n_local_kv_heads, self.head_dim)

        xq=self.q_norm(xq)
        xk = self.k_norm(xk)

        cos,sin=position_embeddings
        xq,xk=apply_rotary_pos_emb(xq,xk,cos[:seq],sin[:seq])

        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        if use_cache:
            present_key_value=(xk,xv)
        else:
            present_key_value=None

        xq,xk,xv=xq.transpose(1,2),repeat_kv(xk,self.n_local_heads//self.n_local_kv_heads).transpose(1,2),repeat_kv(xv,self.n_local_heads//self.n_local_kv_heads).transpose(1,2)

        if self.flash and seq>1 and (not self.is_casual or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output= F.scaled_dot_product_attention(xq,xk,xv,dropout_p=self.dropout if self.training else 0.0,is_causal=self.is_casual)
        else:
            scores=xq@xk.transpose(-1,-2)/math.sqrt(self.head_dim)
            if self.is_casual:
                scores[...,-seq:]+=torch.full((seq,seq),float("-inf"),device=scores.device).triu(1)
            if attention_mask is not None:
                scores+=(1.0-attention_mask.unsqueeze(1).unsqueeze(2))*-1e9
            output=self.atten_dropout(torch.softmax(scores,-1))@xv
        output=output.transpose(1,2).reshape(bsz,seq,-1)
        return self.resid_dropout(output),present_key_value

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size=intermediate_size or config.intermediate_size
        self.up_proj=nn.Linear(config.hidden_size,intermediate_size,bias=False)
        self.act=ACT2FN[config.hidden_act]
        self.gate=nn.Linear(config.hidden_size,intermediate_size,bias=False)
        self.down_proj=nn.Linear(intermediate_size,config.hidden_size,bias=False)
        self.dropout=nn.Dropout(config.dropout)

    def forward(self,x):
        return self.dropout(self.down_proj(self.up_proj(x)*self.act(self.gate(x))))

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super.__init__()
        self.attention=Attention(config)
        self.input_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp=FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False,
                    attention_mask=None):
        residual=hidden_states
        hidden_states,present_key_value=self.attention(self.input_layernorm(hidden_states),position_embeddings,past_key_value,use_cache,attention_mask)
        hidden_states+=residual
        hidden_states=self.mlp(self.post_attention_layernorm(hidden_states))+hidden_states
        return hidden_states,present_key_value
