from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply

class IbcHybridP(BasePolicy):
    """IBC‑style energy model adapted for the DP‑3D framework."""

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        obs_encoder: MultiImageObsEncoder,
        *,
        dropout: float = 0.1,
        train_n_neg: int = 128,
        pred_n_iter: int = 5,
        pred_n_samples: int = 16_384,
        kevin_inference: bool = False,
        andy_train: bool = False,
    ) -> None:
        super().__init__()

        # keep encoder reference for later use
        self.obs_encoder = obs_encoder

        # ---------------- meta ------------------------- #
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "Only 1‑D actions are supported"
        self.action_dim: int = action_shape[0]
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # ---------------- network ---------------------- #
        obs_feature_dim = self.obs_encoder.output_shape()[0]
        self.obs_feature_dim = obs_feature_dim

        in_channels = (
            obs_feature_dim * n_obs_steps + self.action_dim * n_action_steps
        )
        self._build_mlp(in_channels, dropout)

        # ---------------- misc ------------------------- #
        self.normalizer = LinearNormalizer()
        self.train_n_neg = train_n_neg
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.kevin_inference = kevin_inference
        self.andy_train = andy_train

    # ================================================== #
    #                  network utilities                 #
    # ================================================== #
    def _build_mlp(self, in_channels: int, dropout: float):
        mid = 1024
        self.dense0 = nn.Linear(in_channels, mid)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(mid, mid)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(mid, mid)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(mid, mid)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(mid, 1)

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor):
        """Energy of each candidate action.
        obs_feat: (B, To, Do)
        action:   (B, N, Ta, Da)
        returns:  (B, N)
        """
        B, N, Ta, Da = action.shape
        s = obs_feat.reshape(B, 1, -1).expand(-1, N, -1)
        x = torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B * N, -1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        return x.reshape(B, N)

    # ================================================== #
    #                      helpers                       #
    # ================================================== #
    def _safe_uniform(self, low: torch.Tensor, high: torch.Tensor):
        """Return a Uniform distribution guaranteeing low < high element‑wise."""
        eps = 1e-3
        mask = (high - low) < eps
        high = torch.where(mask, low + eps, high)
        return torch.distributions.Uniform(low=low, high=high)

    def get_naction_stats(self):
        """Repeat normaliser stats to action_dim and ensure valid range."""
        stats = self.normalizer["action"].get_output_stats()
        if not stats:
            # normaliser not fitted yet – default to [-1,1]
            base_min = torch.full((1,), -1.0, device=self.device)
            base_max = torch.full((1,), 1.0, device=self.device)
        else:
            base_min = stats["min"].to(self.device)
            base_max = stats["max"].to(self.device)
        reps = self.action_dim // base_min.shape[0]
        low = base_min.repeat(reps)
        high = base_max.repeat(reps)
        # guarantee high > low
        mask = (high - low) < 1e-3
        high[mask] = low[mask] + 1e-3
        return {"min": low, "max": high}

    # ================================================== #
    #                    inference                       #
    # ================================================== #
    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]):
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        device, dtype = self.device, self.dtype

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps].reshape(-1, *x.shape[2:])
        )
        obs_feat = self.obs_encoder(this_nobs).reshape(B, self.n_obs_steps, -1)

        stats = self.get_naction_stats()
        action_dist = self._safe_uniform(stats["min"], stats["max"])
        samples = action_dist.sample((B, self.pred_n_samples, self.n_action_steps)).to(
            device=device, dtype=dtype
        )

        zero = torch.tensor(0.0, device=device)
        resample_std = torch.tensor(3e-2, device=device)
        for i in range(self.pred_n_iter):
            logits = self.forward(obs_feat, samples)
            prob = torch.softmax(logits, dim=-1)
            if i < self.pred_n_iter - 1:
                idx = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(B, device=device).unsqueeze(-1), idx]
                samples += torch.normal(zero, resample_std, size=samples.shape, device=device)
        idx = torch.multinomial(prob, 1)
        best = samples[torch.arange(B, device=device).unsqueeze(-1), idx].squeeze(1)
        action = self.normalizer["action"].unnormalize(best)
        return {"action": action}

    # ================================================== #
    #                     training                       #
    # ================================================== #
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])
        B = naction.shape[0]
        device, dtype = self.device, self.dtype

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps].reshape(-1, *x.shape[2:])
        )
        obs_feat = self.obs_encoder(this_nobs).reshape(B, self.n_obs_steps, -1)

        # positives
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        pos = naction[:, start:end]
        pos += torch.normal(0, 1e-4, size=pos.shape, device=device, dtype=dtype)

        # negatives
        stats = self.get_naction_stats()
        action_dist = self._safe_uniform(stats["min"], stats["max"])
        neg = action_dist.sample((B, self.train_n_neg, self.n_action_steps)).to(device=device, dtype=dtype)
        samples = torch.cat([pos.unsqueeze(1), neg], dim=1)

        if self.andy_train:
            logits = self.forward(obs_feat, samples).log_softmax(dim=-1)
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1
            loss = -(logits * labels).sum(-1).mean()
        else:
            logits = self.forward(obs_feat, samples)
            target = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
        return loss, {"energy_ce": float(loss.detach())}
