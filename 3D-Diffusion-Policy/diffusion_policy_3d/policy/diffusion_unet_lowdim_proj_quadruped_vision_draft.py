"""diffusion_unet_lowdim_proj_quadruped.py
================================================
Pure **yaw‑compensation** diffusion policy for the quadruped‑walk task.

This policy injects a configurable yaw‑velocity offset into the `yaw_index`
action dimension when the external IMU roll (Euler `rpy0`) drifts outside a
small dead‑band.

The `projection()` is called automatically from `conditional_sample()` so all
returned trajectories are already compensated.
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DiffusionUnetLowdimProjQuadrupedVision(BasePolicy):
    """Low‑dim diffusion policy with **vision guidance with aruco detection** only."""

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        *,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        # yaw‑compensation knobs
        additional_yaw_gain: float = 0.001,  # rad/s added when |rpy0|>thr
        rpy_threshold: float = 0.035,        # rad – dead‑band threshold
        yaw_index: int = 4,                 # which action dim is yaw vel
        # standard diffusion policy knobs
        num_inference_steps: Optional[int] = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        # --- store yaw compensation knobs ---
        self.additional_yaw_gain = additional_yaw_gain
        self.rpy_threshold = rpy_threshold
        self.yaw_index = yaw_index

        # --- standard diffusion policy init ---
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ------------------------------------------------------------------
    # Inference with built‑in projection
    # ------------------------------------------------------------------
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        *,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        # special arg for this policy
        imu_euler: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Runs DDPM sampling and applies yaw compensation after each step."""
        model = self.model
        scheduler = self.noise_scheduler

        # Patch for torch <1.7: generator argument not supported in randn_like
        if generator is not None:
            trajectory = torch.randn(condition_data.shape, dtype=condition_data.dtype, device=condition_data.device, generator=generator)
        else:
            trajectory = torch.randn_like(condition_data)
        scheduler.set_timesteps(self.num_inference_steps)

        inference_step = 0
        for t in scheduler.timesteps:
            # enforce conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # predict noise
            model_out = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # compute previous sample
            trajectory = scheduler.step(model_out, t, trajectory, **kwargs).prev_sample

            # --- yaw compensation on un-normalised actions ---
            # (un-normalize, project, re-normalize)
            if inference_step >= 7:
                naction = trajectory[..., :self.action_dim]
                action = self.normalizer['action'].unnormalize(naction)
                action = self.projection(action, imu_euler)
                trajectory[..., :self.action_dim] = self.normalizer['action'].normalize(action)
            inference_step += 1 
        # final conditioning
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def projection(self, x: torch.Tensor, imu_euler: float) -> torch.Tensor:
        """Add yaw offset if |imu_euler| > threshold.""" 
        # rpy0 euler angle reduces when the robot is pulled by human and incline on the left side
        # yaw speed reduces if robot rotate cloclwise from bird-eye view
        # Let's print the value every time to check
        # print(f"Projection called with imu_euler: {imu_euler:.4f}, threshold: {self.rpy_threshold:.4f}")
        # if abs(imu_euler) <= self.rpy_threshold:
        #     print("Projection not activated: imu_euler within threshold.")
        #     return x
        # print("Projection activated: imu_euler exceeds threshold.")
        # if abs(imu_euler) <= self.rpy_threshold:
        #     return x
        yaw_offset = 0.0
        if imu_euler > self.rpy_threshold:
            yaw_offset = -self.additional_yaw_gain
            print("euler angle larger than threshold, guidance on the right side")
        elif imu_euler < -self.rpy_threshold:
            yaw_offset = self.additional_yaw_gain
            print("euler angle smaller than negative threshold, guidance on the left side")
        else:
            return x
        x = x.clone()
        x[..., self.yaw_index] += yaw_offset
        # print(f"x after projection:{x}")
        return x

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key.
        kwargs: must include "imu_euler" for this policy.
        result: must include "action" key.
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict
        assert 'imu_euler' in kwargs, "imu_euler must be provided for projection"

        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros((B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs
            shape = (B, T, Da)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs.reshape(B, -1)
            shape = (B, self.n_action_steps if self.pred_action_steps_only else T, Da)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs
            cond_mask[:, :To, Da:] = True

        # run sampling with projection
        nsample = self.conditional_sample(
            cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond, **kwargs
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To if not self.oa_step_convention else To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {'action': action, 'action_pred': action_pred}
        return result

    # ------------------------------------------------------------------

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        B = action.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_dim
        To = self.n_obs_steps
        
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:To,:].reshape(B, -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict

