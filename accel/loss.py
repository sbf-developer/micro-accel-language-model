"""Fisher-inspired and focal token weighting for sample-efficient training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fisher_weighted_loss(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 0.5,
    cap: float = 3.0,
) -> torch.Tensor:
    """
    Approximate Fisher information gain weighting (FisherSFT-inspired).

    High-surprisal tokens get upweighted because they carry more information
    about the gradient direction. We use detached per-token loss as a cheap
    Fisher proxy without full Hessian computation.
    """
    masked = per_token_loss * loss_mask
    active = loss_mask > 0
    if active.sum() == 0:
        return per_token_loss.sum() * 0.0

    mean_loss = masked[active].mean().detach().clamp(min=1e-6)
    weights = (masked.detach() / mean_loss).clamp(0.25, cap).pow(gamma)
    weighted = masked * weights
    return weighted.sum() / loss_mask.sum().clamp(min=1.0)


def focal_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Focal loss variant: down-weight easy tokens, focus learning budget."""
    b, s, v = logits.shape
    ce = F.cross_entropy(
        logits.view(-1, v),
        targets.view(-1),
        reduction="none",
    ).view(b, s)
    pt = torch.exp(-ce.detach())
    focal = ((1 - pt) ** gamma) * ce
    focal = focal * loss_mask
    loss = focal.sum() / loss_mask.sum().clamp(min=1.0)
    return loss, ce
