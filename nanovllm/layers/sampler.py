import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor):
        # Apply temperature scaling
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        
        # Apply top-p (nucleus) sampling for samples where top_p < 1.0
        # Shape: probs is (batch_size, vocab_size)
        needs_top_p = top_ps < 1.0
        
        if needs_top_p.any():
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            # Compute cumulative probabilities
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            # Create mask: keep tokens until cumulative prob exceeds top_p
            # cumsum_probs - sorted_probs gives the cumulative prob before adding current token
            mask = cumsum_probs - sorted_probs > top_ps.unsqueeze(dim=1)
            # Zero out probabilities beyond the top-p threshold
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            # Renormalize
            sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
            # Scatter back to original indices
            probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
        
        # Sample using Gumbel-Max trick
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
