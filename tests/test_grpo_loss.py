"""Unit tests for GRPO loss and group normalization."""

import torch
import pytest

from qwen3_rl.loss.grpo import grpo_loss, compute_group_advantages


class TestComputeGroupAdvantages:

    def test_normal_case(self):
        adv = compute_group_advantages([1.0, 0.0, 1.0, 0.0])
        assert adv is not None
        assert len(adv) == 4
        assert abs(sum(adv) / len(adv)) < 0.1  # mean ~ 0

    def test_degenerate_all_same(self):
        assert compute_group_advantages([1.0, 1.0, 1.0, 1.0]) is None

    def test_degenerate_all_zero(self):
        assert compute_group_advantages([0.0, 0.0, 0.0]) is None

    def test_ordering(self):
        adv = compute_group_advantages([0.0, 0.5, 1.0])
        assert adv is not None
        assert adv[0] < adv[1] < adv[2]

    def test_negative_rewards(self):
        adv = compute_group_advantages([-1.0, 0.0, 1.0, 2.0])
        assert adv is not None
        assert adv[0] < 0 < adv[-1]


class TestGRPOLoss:

    def _make_tensors(self, S=10, trainable_frac=0.6):
        logp = torch.randn(S, dtype=torch.float32)
        mask = torch.zeros(S, dtype=torch.bool)
        n_train = int(S * trainable_frac)
        mask[S - n_train:] = True
        return logp, mask

    def test_basic_no_nan(self):
        logp_new, mask = self._make_tensors()
        logp_old = logp_new.clone()
        logp_ref = logp_new.clone() + 0.1

        loss, metrics = grpo_loss(logp_new, logp_old, logp_ref, mask, advantage=1.0)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert metrics["numerical_clamp_count"] == 0

    def test_ratio_near_one_at_start(self):
        logp, mask = self._make_tensors()
        loss, metrics = grpo_loss(logp, logp.clone(), logp.clone(), mask, advantage=1.0)
        assert abs(metrics["mean_ratio"] - 1.0) < 1e-5

    def test_positive_advantage_negative_loss(self):
        logp, mask = self._make_tensors()
        loss_pos, _ = grpo_loss(logp, logp.clone(), logp.clone(), mask, advantage=1.0)
        loss_neg, _ = grpo_loss(logp, logp.clone(), logp.clone(), mask, advantage=-1.0)
        # with ratio=1, loss = -min(1*A, clip(1)*A) = -A
        assert loss_pos.item() < 0
        assert loss_neg.item() > 0

    def test_clipping_behavior(self):
        logp_old, mask = self._make_tensors(S=20)
        logp_new = logp_old + 0.5  # ratio = exp(0.5) ≈ 1.65, > 1+eps_high=1.28

        loss, metrics = grpo_loss(
            logp_new, logp_old, logp_old, mask, advantage=1.0,
            eps_low=0.2, eps_high=0.28,
        )
        assert metrics["clip_fraction"] > 0

    def test_numerical_clamp_fires(self):
        logp_old, mask = self._make_tensors()
        logp_new = logp_old + 25.0  # exceeds ratio_clamp=20

        loss, metrics = grpo_loss(
            logp_new, logp_old, logp_old, mask, advantage=1.0,
            ratio_clamp=20.0,
        )
        assert metrics["numerical_clamp_count"] > 0
        assert not torch.isnan(loss)

    def test_kl_penalty(self):
        logp, mask = self._make_tensors()
        logp_ref = logp + 0.1

        loss_no_kl, m1 = grpo_loss(logp, logp.clone(), logp_ref, mask, advantage=1.0, beta_kl=0.0)
        loss_kl, m2 = grpo_loss(logp, logp.clone(), logp_ref, mask, advantage=1.0, beta_kl=0.1)

        assert m1["approx_kl"] == pytest.approx(0.0, abs=1e-5)
        assert abs(loss_kl.item() - loss_no_kl.item()) > 1e-6

    def test_empty_mask(self):
        S = 10
        logp = torch.randn(S)
        mask = torch.zeros(S, dtype=torch.bool)
        loss, metrics = grpo_loss(logp, logp.clone(), logp.clone(), mask, advantage=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_masked_reduction_matches_naive(self):
        S = 8
        logp_old = torch.randn(S)
        logp_new = logp_old + 0.1
        logp_ref = logp_old - 0.05
        mask = torch.tensor([False, False, True, True, True, False, True, False])
        advantage = 0.5

        loss, _ = grpo_loss(logp_new, logp_old, logp_ref, mask, advantage)

        # manual computation
        log_ratio = torch.clamp(logp_new - logp_old, -20, 20)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 0.8, 1.28)
        per_token = -torch.min(ratio * advantage, clipped * advantage)
        expected = per_token[mask].mean()

        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)
