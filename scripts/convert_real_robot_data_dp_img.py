import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time
import socket
import pickle
import open3d as o3d
import cv2

def preproces_image(image):
    img_size = 84
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx3 -> 3xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 3xHxW -> HxWx3
    image = image.cpu().numpy()
    return image

expert_data_path = '/home/yxt/thesis/yirui/imitation/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1-12-complete/pick-place-kinect-rs'
save_data_path = '/home/yxt/thesis/yirui/imitation/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1-12-complete/pick-place-kinect-rs-dp-img.zarr'
demo_dirs = [os.path.join(expert_data_path, f) for f in os.listdir(expert_data_path) if f.endswith('.pkl')]

print(f"demo_dirs: {len(demo_dirs)}")

# storage
total_count = 0
point_cloud_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []
rs_arrays = []
azure_arrays = []

if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

for demo_dir in demo_dirs:
    dir_name = os.path.dirname(demo_dir)

    cprint('Processing {}'.format(demo_dir), 'green')
    with open(demo_dir, 'rb') as f:
        demo = pickle.load(f)

    pcd_dirs = os.path.join(dir_name, 'pcd')
    if not os.path.exists(pcd_dirs):
           os.makedirs(pcd_dirs)
        
    demo_length = len(demo['action'])
    for step_idx in tqdm.tqdm(range(demo_length)):
       
        total_count += 1
        rs_image = demo['rs_img'][step_idx]
        rs_image = preproces_image(rs_image)
        azure_image = demo['azure_rgb'][step_idx]
        azure_image = preproces_image(azure_image)
        robot_state = demo['agent_pose'][step_idx]
        action = demo['action'][step_idx]
        if action[6] == 1:
            action = action[:6]
            action = np.append(action, [1, 0])
        elif action[6] == 0:
            action = action[:6]
            action = np.append(action, [0, 1])

        rs_arrays.append(rs_image)
        azure_arrays.append(azure_image)
        action_arrays.append(action)
        state_arrays.append(robot_state)
    
    episode_ends_arrays.append(total_count)

# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

rs_arrays = np.stack(rs_arrays, axis=0)
if rs_arrays.shape[1] == 3: # make channel last
    rs_arrays = np.transpose(rs_arrays, (0,2,3,1))
azure_arrays = np.stack(azure_arrays, axis=0)
if azure_arrays.shape[1] == 3: # make channel last
    azure_arrays = np.transpose(azure_arrays, (0,2,3,1))
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
rs_chunk_size = (100, rs_arrays.shape[1], rs_arrays.shape[2], rs_arrays.shape[3])
azure_chunk_size = (100, azure_arrays.shape[1], azure_arrays.shape[2], azure_arrays.shape[3])
if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError
zarr_data.create_dataset('rs_img', data=rs_arrays, chunks=rs_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('azure_img', data=azure_arrays, chunks=azure_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
cprint(f'rs_img shape: {rs_arrays.shape}, range: [{np.min(rs_arrays)}, {np.max(rs_arrays)}]', 'green')
cprint(f'azure_img shape: {azure_arrays.shape}, range: [{np.min(azure_arrays)}, {np.max(azure_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

