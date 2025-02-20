import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import yaml

from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ, PointNetEncoderXYZRGB, create_mlp
from diffusion_policy_3d.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy_3d.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy_3d.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy_3d.model.vision.model_getter import get_resnet

class MultiModalEncoder(nn.Module):
    def __init__(self, 
                 shape_meta: dict,
                 rgb_model: Union[nn.Module, Dict[str,nn.Module]],
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
                 crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
                 random_crop: bool=True,
                 # replace BatchNorm with GroupNorm
                 use_group_norm: bool=False,
                 # renormalize rgb input with imagenet normalization
                 # assuming input in [0,1]
                 imagenet_norm: bool=False
                ):
        super().__init__()
        self.point_cloud_key = 'point_cloud'
        self.state_key = 'agent_pos'
        self.img_key = 'certainty_map'
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict()
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_dict[key] = shape

        self.point_cloud_shape = obs_dict[self.point_cloud_key]
        self.state_shape = obs_dict[self.state_key]
        self.img_shape = obs_dict[self.img_key]
        self.obs_dict = obs_dict

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.img_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        self.rgb_model = rgb_model
        if self.rgb_model is not None:
            if use_group_norm:
                        self.rgb_model = replace_submodules(
                            root_module=self.rgb_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )

        # configure resize
        input_shape = self.img_shape
        this_resizer = nn.Identity()
        if resize_shape is not None:
            h, w = resize_shape
            this_resizer = torchvision.transforms.Resize(
                size=(h,w)
            )
            input_shape = (self.img_shape[0],h,w)

        # configure randomizer
        this_randomizer = nn.Identity()
        if crop_shape is not None:
            h, w = crop_shape
            if random_crop:
                this_randomizer = CropRandomizer(
                    input_shape=input_shape,
                    crop_height=h,
                    crop_width=w,
                    num_crops=1,
                    pos_enc=False
                )
            else:
                this_normalizer = torchvision.transforms.CenterCrop(
                    size=(h,w)
                )

        # configure normalizer
        this_normalizer = nn.Identity()
        if imagenet_norm:
            this_normalizer = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.img_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        points_feature = self.extractor(points)

        state = observations[self.state_key]
        state_feat = self.state_mlp(state) 

        image = observations[self.img_key]
        image = image.reshape(-1, *self.img_shape)
        image = self.img_transform(image)
        image_feat = self.rgb_model(image)

        feature = torch.cat([points_feature, state_feat, image_feat], dim=1)
        return feature
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_dict = self.obs_dict
        batch_size = 1
        for key, attr in obs_dict.items():
            shape = tuple(attr)
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=torch.float32,
                device='cpu')
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
    