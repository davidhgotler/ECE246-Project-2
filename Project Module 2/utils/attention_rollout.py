import torch

def attention_rollout(attentions):
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1) 
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) 
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(rollout, attention_heads_fused) 

    return rollout