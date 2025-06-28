from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.multi_modal_obs_encoder import MultiModalEncoder

import casadi as ca
import numpy as np

def custom_loss(pred, target, mask):
    pred_regression = pred[:, :, :6]
    target_regression = target[:, :, :6]

    pred_classification = pred[:, :, 6:]
    target_classification = target[:, :, 6:]
    target_classification = torch.argmax(target_classification, dim=-1)

    mse_loss = F.mse_loss(pred_regression, target_regression, reduction='none')
    mse_loss = mse_loss * mask[:, :, :6].type(mse_loss.dtype)
    mse_loss = reduce(mse_loss, 'b ... -> b', 'mean')
    mse_loss = mse_loss.mean()
    cross_entropy_loss = F.cross_entropy(pred_classification.permute(0, 2, 1), target_classification, reduction='none')
    cross_entropy_loss = cross_entropy_loss * mask[:, :, 0].type(cross_entropy_loss.dtype)
    cross_entropy_loss = reduce(cross_entropy_loss, 'b ... -> b', 'mean')
    cross_entropy_loss = cross_entropy_loss.mean()
    weight_mse = 1.0
    weight_ce = 1.0
    loss = weight_mse * mse_loss + weight_ce * cross_entropy_loss
    return loss

class SimpleDP3MMProjFlag(BasePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            multi_modal_encoder: MultiModalEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            end_effector_length,
            adapt_height,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            encoder_output_dim = 256,
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            use_pc_color=False,
            pointnet_type="pointnet",
            train_gripper_length = 0.099,
            **kwargs):
        super().__init__()

        self.adapt_height = adapt_height
        self.end_effector_length = end_effector_length
        self.condition_type = condition_type
        self.train_gripper_length = train_gripper_length

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # create diffusion model
        obs_feature_dim = multi_modal_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[SDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[SDP3] pointnet_type: {self.pointnet_type}", "yellow")
        cprint(f"current gripper length: {self.end_effector_length} m", "yellow")
        cprint(f"train gripper length: {train_gripper_length} m", "yellow")
        cprint(f"adapt height: {self.adapt_height}", "yellow")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = multi_modal_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.previous_adopt_flag = None


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None, obs_dict=None, Da=None, start=None, end=None, adopt_flag=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        pose = obs_dict['agent_pos'][:, -1, :]

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        infernce_step = 0
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. sample from model
            model_output = model(sample=trajectory,
                                timestep=t,
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
            # 4. apply projection
            if infernce_step >= 5:
                # Unnormalize the action sequence
                naction_pred = trajectory[..., :Da]
                action_pred = self.normalizer['action'].unnormalize(naction_pred)
                action = action_pred[:, start:end]
                
                # Apply projection in the unnormalized space
                action = self.projection(pose, action, condition_data.device, adopt_flag)
                action_pred[:, start:end] = action
                
                # Normalize the action sequence again
                trajectory[..., :Da] = self.normalizer['action'].normalize(action_pred)
            
            infernce_step += 1

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask] 


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        bs = obs_dict['agent_pos'].shape[0]
        self.adopt_flag = torch.zeros(size=(bs, 1), device=self.device, dtype=torch.bool)

        # Define the rectangle bounds
        x_min, y_min = 0.48, -0.134
        x_max, y_max = 0.69, 0.0473
        z_max = 0.1

        for i in range(bs):
            x, y, z = obs_dict['agent_pos'][i, -1, 0], obs_dict['agent_pos'][i, -1, 1], obs_dict['agent_pos'][i, -1, 2] # x and y at the last timestep
            if x_min <= x <= x_max and y_min <= y <= y_max and z - self.end_effector_length < z_max:
                self.adopt_flag[i] = True
        
        print("adopt_flag:", self.adopt_flag)
        
        self.adopt_flag = self.adopt_flag.squeeze(-1)
        
        if self.adapt_height:
            obs_dict['agent_pos'][:, :, 2] = obs_dict['agent_pos'][:, :, 2] - self.end_effector_length
        else:
            # Gradual change logic
            if self.previous_adopt_flag is None:
                self.previous_adopt_flag = torch.zeros_like(self.adopt_flag)

            for i in range(bs):
                if self.adopt_flag[i] and not self.previous_adopt_flag[i]:
                    # Gradual change from False to True
                    for t in range(obs_dict['agent_pos'].shape[1]):
                        obs_dict['agent_pos'][i, t, 2] -= (self.end_effector_length - self.train_gripper_length) * ((t + 1) / (obs_dict['agent_pos'].shape[1]))
                elif not self.adopt_flag[i] and self.previous_adopt_flag[i]:
                    # Gradual change from True to False
                    for t in range(obs_dict['agent_pos'].shape[1]):
                        obs_dict['agent_pos'][i, t, 2] -= (self.end_effector_length - self.train_gripper_length) * ((obs_dict['agent_pos'].shape[1] - t - 1) / (obs_dict['agent_pos'].shape[1]))
                elif self.adopt_flag[i] and self.previous_adopt_flag[i]:
                    # Keep the height constant
                    obs_dict['agent_pos'][i, :, 2] = obs_dict['agent_pos'][i, :, 2] - (self.end_effector_length - self.train_gripper_length)

            self.previous_adopt_flag = self.adopt_flag.clone()

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        start = To - 1
        end = start + self.n_action_steps

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            obs_dict=obs_dict, Da=Da, start=start, end=end, adopt_flag=self.adopt_flag,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result, self.adopt_flag

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        if self.adapt_height:
            batch['obs']['agent_pos'][:, :, 2] = batch['obs']['agent_pos'][:, :, 2] - self.end_effector_length
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
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
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        #loss = custom_loss(pred, target, loss_mask)
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        loss_dict = {
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict
    
    def projection(self, pose, x_init, device, adopt_flag):
        """
        x_init: Tensor of shape (B, T, D_a)
        pose: Tensor of shape (B, D_p) where D_p >= 3
        """
        if isinstance(x_init, torch.Tensor):
            x_init_np = x_init.detach().cpu().numpy()
            torch_dtype = x_init.dtype
        else:
            x_init_np = x_init
            torch_dtype = numpy_to_torch_dtype(x_init.dtype)

        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        
        B, T, D_a = x_init_np.shape
        
        x_proj = np.zeros_like(x_init_np)
        
        for b in range(B):
            # Initialize cumulative_pose for the batch
            cumulative_pose = pose[b, :].copy()

            # if self.end_effector_length >= self.train_gripper_length:
            x_var = ca.SX.sym('x', T)
            x_init_dm = ca.DM(x_init_np[b, :, 2])  # Extract the third element for all timesteps
            
            ## Objective function: minimize the squared L2 norm between x_var and x_init_dm
            obj = ca.sumsqr(x_var - x_init_dm)
            
            ## Constraints for the entire sequence
            if self.adapt_height:
                g1 = cumulative_pose[2] + ca.sum1(x_var)
            else:
                if adopt_flag[b] == True:
                    g1 = cumulative_pose[2] + ca.sum1(x_var) -  self.train_gripper_length
                else:    
                    g1 = cumulative_pose[2] + ca.sum1(x_var) - self.end_effector_length
            
            ## Ensure the sum of x_var has the same sign as the sum of x_init_dm
            sum_x_init_dm = ca.sum1(x_init_dm)
            sum_x_var = ca.sum1(x_var)
            
            if sum_x_init_dm >= 0:
                g2 = sum_x_var
                lbg2 = 0
                ubg2 = ca.inf
            else:
                g2 = -sum_x_var
                lbg2 = 0
                ubg2 = ca.inf
            
            nlp = {'x': x_var, 'f': obj, 'g': ca.vertcat(g1, g2)}
            # nlp = {'x': x_var, 'f': obj, 'g': ca.vertcat(g1)}

            ## Suppress CasADi output by setting solver options
            solver = ca.nlpsol(
                'solver', 'ipopt', nlp,
                {'print_time': False, 'ipopt': {'print_level': 0}}
            )
            
            lbg = ca.vertcat(0.01, lbg2)   # Lower bound for g1 and g2
            ubg = ca.vertcat(ca.inf, ubg2)  # Upper bound for g1 and g2
            
            ## Solve the problem
            sol = solver(x0=x_init_dm, lbg=lbg, ubg=ubg)
            x_proj_vars = sol['x'].full().flatten()
            
            print(f"Initial x: {x_init_dm}")
            print(f"Projected x: {x_proj_vars}")
            ## Reconstruct full x_proj
            x_proj[b, :, :] = x_init_np[b, :, :]
            x_proj[b, :, 2] = x_proj_vars
        
        x_proj_tensor = torch.from_numpy(x_proj).to(torch_dtype).to(device)        
        return x_proj_tensor
    
def numpy_to_torch_dtype(np_dtype):
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    else:
        raise TypeError(f"Unsupported NumPy dtype: {np_dtype}")

