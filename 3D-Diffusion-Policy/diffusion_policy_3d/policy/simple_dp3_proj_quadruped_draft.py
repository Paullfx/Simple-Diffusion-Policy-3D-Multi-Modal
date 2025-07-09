from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import casadi as ca
import numpy as np

# Optional ROS2 imports – only required when using the *no‑QP* projection mode
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from std_msgs.msg import Float32  # type: ignore
except ImportError:  # pragma: no cover – ROS2 not always installed in every env
    rclpy = None  # type: ignore

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator


# ============================================================================
# Utility helpers
# ============================================================================

def numpy_to_torch_dtype(np_dtype):
    """Map NumPy dtypes to the equivalent PyTorch dtype."""
    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.float64:
        return torch.float64
    if np_dtype == np.int32:
        return torch.int32
    if np_dtype == np.int64:
        return torch.int64
    raise TypeError(f"Unsupported NumPy dtype: {np_dtype}")


# ============================================================================
# Policy definition
# ============================================================================

class DiffusionUnetLowdimPolicy(BasePolicy):
    """Low‑dim diffusion policy **with optional projection**.

    Besides the original QP‑based projection (to enforce a minimum height
    clearance), we now support a *no‑QP* mode that reacts to the robot’s IMU
    roll/pitch measurement published on the ROS 2 topic `/sport/imu/rpy0`.

    If the roll angle (``rpy0``) drifts outside ±0.02 rad while the model is
    evaluated, an additional yaw velocity offset is injected in the action’s
    **fourth** dimension to stabilise the quadruped’s heading.
    """

    # ---------------------------------------------------------------------
    # ROS helper node (only created when use_qp=False)
    # ---------------------------------------------------------------------
    class _ImuSubscriber(Node):
        def __init__(self):
            super().__init__('dp3_policy_imu_sub')
            self.rpy0: float = 0.0
            self.create_subscription(Float32, '/sport/imu/rpy0', self.cb, 10)

        def cb(self, msg: Float32):  # noqa: D401 – simple callback
            self.rpy0 = float(msg.data)

    # ------------------------------------------------------------------
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        *,
        # ------------------ projection‑specific kwargs ------------------
        use_qp: bool = False,
        additional_yaw_gain: float = 0.15,  # rad / s injected when rpy0≫0
        rpy_threshold: float = 0.02,  # trigger threshold (rad)
        # ---------------------------------------------------------------
        num_inference_steps: Optional[int] = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        # all other kwargs forwarded to BasePolicy.step etc.
        **kwargs,
    ) -> None:
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        # ------------------------------ store args ---------------------
        self.use_qp = use_qp
        self.additional_yaw_gain = additional_yaw_gain
        self.rpy_threshold = rpy_threshold
        self._imu_node: Optional[DiffusionUnetLowdimPolicy._ImuSubscriber] = None

        # ----------------------- original initialisation ---------------
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

        # --------------------- init ROS2 subscriber if needed ----------
        if not self.use_qp:
            if rclpy is None:
                raise ImportError(
                    "rclpy is required for use_qp=False but is not installed in this environment."
                )
            if not rclpy.ok():
                rclpy.init(args=None)
            self._imu_node = DiffusionUnetLowdimPolicy._ImuSubscriber()

    # ==================================================================
    # Inference helpers (same as original except ROS spin_once)
    # ==================================================================
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        *,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps)

        inference_step = 0
        for t in scheduler.timesteps:

            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory, timestep=t, local_cond=local_cond, global_cond=global_cond)

            trajectory = scheduler.step(model_output, t, trajectory, **kwargs).prev_sample
            # apply projection
            if inference_step >= 7:
                # Unnormalize the action sequence
                naction_pred = trajectory[..., :Da]
                action_pred = self.normalizer['action'].unnormalize(naction_pred)
                action = action_pred[:, start:end]
                # Apply projection in the unnormalized space
                action = self.projection(pose, action, condition_data.device, adopt_flag)
                action_pred[:, start:end] = action
                # Normalize the action sequence again
                trajectory[..., :Da] = self.normalizer['action'].normalize(action_pred)

            inference_step += 1

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ==================================================================
    # Training (unchanged)
    # ==================================================================
    def set_normalizer(self, normalizer: LinearNormalizer) -> None:  # noqa: D401
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):  # identical to original – removed for brevity
        # (… existing implementation …)
        raise NotImplementedError("Training path left unchanged – use the original implementation.")

    # ==================================================================
    # PROJECTION – two cases: no‑QP (IMU‑based) vs. QP (original)
    # ==================================================================
    def projection(
        self,
        pose: torch.Tensor,
        x_init: torch.Tensor,
        device: torch.device | torch.dtype | str,
        adopt_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """QP or **heading compensation**.

        * **No‑QP mode** (`self.use_qp == False`)
            * Reads the latest roll value (`rpy0`) from `/sport/imu/rpy0` via a
              background ROS 2 subscription.
            * If |rpy0| > ``self.rpy_threshold`` (≈ 2 centidegrees), injects an
              extra yaw velocity (sign based on the roll) into the fourth
              action dimension *for all time‑steps*.
            * Otherwise returns the original action unchanged.
        * **QP mode** (`self.use_qp == True`)
            * Runs the original CasADi optimisation to enforce a minimum height
              clearance (see `SimpleDP3ProjFlag.projection`).
        """
        # ----------------------------------------------------------
        # Case 1 – *no‑QP* : react to IMU roll (rpy0)
        # ----------------------------------------------------------
        if not self.use_qp:
            assert self._imu_node is not None, "IMU subscriber not initialised."
            # Process at most one queued message (non‑blocking)
            rclpy.spin_once(self._imu_node, timeout_sec=0.0)
            rpy0_value: float = self._imu_node.rpy0

            if abs(rpy0_value) <= self.rpy_threshold:
                return x_init  # within tolerance → no correction

            # Decide yaw offset direction
            yaw_offset = self.additional_yaw_gain if rpy0_value > 0 else -self.additional_yaw_gain

            x_corrected = x_init.clone()
            # Quadruped‑walk action layout: dim‑3 ≡ yaw velocity
            x_corrected[..., 3] += yaw_offset
            return x_corrected

        # ----------------------------------------------------------
        # Case 2 – *with QP* : original optimisation
        # ----------------------------------------------------------
        # Prepare defaults & converts
        if adopt_flag is None:
            adopt_flag = torch.zeros(x_init.shape[0], dtype=torch.bool, device=x_init.device)

        if isinstance(x_init, torch.Tensor):
            x_init_np = x_init.detach().cpu().numpy()
            torch_dtype = x_init.dtype
        else:
            x_init_np = x_init
            torch_dtype = numpy_to_torch_dtype(x_init.dtype)

        pose_np = pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else pose
        B, T, D_a = x_init_np.shape
        x_proj = np.zeros_like(x_init_np)

        for b in range(B):
            cumulative_z = pose_np[b, 2].copy()
            x_proj_vars = np.zeros(T)

            for t in range(T):
                x_var = ca.SX.sym('x', 1)
                x_init_scalar = ca.DM([x_init_np[b, t, 2]])
                obj = ca.sumsqr(x_var - x_init_scalar)

                if self.adapt_height:
                    g_t = cumulative_z + x_var  # dynamic baseline
                else:
                    clearance = (
                        self.train_gripper_length if adopt_flag[b].item() else self.end_effector_length
                    )
                    g_t = cumulative_z + x_var - clearance

                nlp = {'x': x_var, 'f': obj, 'g': g_t}
                solver = ca.nlpsol('solver', 'ipopt', nlp, {'print_time': False, 'ipopt': {'print_level': 0}})
                sol = solver(x0=x_init_scalar, lbg=[0.01], ubg=[ca.inf])
                x_proj_vars[t] = sol['x'].full().flatten()[0]
                cumulative_z += x_proj_vars[t]

            # overwrite z‑component, keep others intact
            x_proj[b] = x_init_np[b]
            x_proj[b, :, 2] = x_proj_vars

        return torch.from_numpy(x_proj).to(torch_dtype).to(device)
