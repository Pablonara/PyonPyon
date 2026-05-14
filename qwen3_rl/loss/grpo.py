from __future__ import annotations

import warnings

import torch


def compute_group_advantages(
    rewards: list[float],
    threshold: float = 1e-6,
) -> list[float] | None:
    t = torch.tensor(rewards, dtype=torch.float64)
    std = t.std().item()
    if std < threshold:
        return None
    mean = t.mean().item()
    return [float((r - mean) / (std + 1e-4)) for r in rewards]


_clamp_warned = False


def grpo_loss(
    logp_new: torch.FloatTensor,
    logp_old: torch.FloatTensor,
    logp_ref: torch.FloatTensor,
    mask: torch.BoolTensor,
    advantage: float,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    beta_kl: float = 0.0,
    ratio_clamp: float = 20.0,
) -> tuple[torch.Tensor, dict]:
    global _clamp_warned

    mask_f = mask.float()
    n_trainable = mask_f.sum().clamp(min=1.0)

    log_ratio = logp_new - logp_old
    clamped = (log_ratio.abs() > ratio_clamp) & mask
    numerical_clamp_count = int(clamped.sum().item())
    if numerical_clamp_count > 0 and not _clamp_warned:
        warnings.warn(
            f"grpo_loss: {numerical_clamp_count} tokens hit ratio_clamp={ratio_clamp}. "
            f"Investigate logp drift (kernel divergence, stale logp_old, dtype mismatch).",
            RuntimeWarning,
            stacklevel=2,
        )
        _clamp_warned = True

    log_ratio = torch.clamp(log_ratio, -ratio_clamp, ratio_clamp)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)

    loss_unclipped = ratio * advantage
    loss_clipped = clipped_ratio * advantage
    loss_per_token = -torch.min(loss_unclipped, loss_clipped)

    loss = (loss_per_token * mask_f).sum() / n_trainable

    # KL penalty (reverse KL estimator)
    if beta_kl > 0.0:
        kl_per_token = torch.exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1.0
        kl = (kl_per_token * mask_f).sum() / n_trainable
        loss = loss + beta_kl * kl
    else:
        kl_per_token = torch.zeros_like(logp_new)
        kl = torch.tensor(0.0, device=logp_new.device)

    clip_fraction = float(
        ((ratio < 1.0 - eps_low) | (ratio > 1.0 + eps_high)).float()[mask].mean().item()
    ) if mask.any() else 0.0

    metrics = {
        "clip_fraction": clip_fraction,
        "numerical_clamp_count": numerical_clamp_count,
        "approx_kl": float((kl_per_token * mask_f).sum().item() / n_trainable.item()),
        "mean_ratio": float(ratio[mask].mean().item()) if mask.any() else 1.0,
        "max_ratio": float(ratio[mask].max().item()) if mask.any() else 1.0,
    }
    return loss, metrics
