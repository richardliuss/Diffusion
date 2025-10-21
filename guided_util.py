import argparse
import os
import time

import numpy as np
import torch_npu
import torch as th
import blobfile as bf
import torch.distributed as dist
import torch.nn.functional as F


def get_guided_arr_dict(path, in_channels):
    guided_arr_dict = {}
    for file in bf.listdir(path):
        # file only contains the name of the file, not the full path
        if file.endswith(".npy"):
            with bf.BlobFile(os.path.join(path,file), "rb") as f:
                # array [180, 360, depth_level] HWC and rescale to [-1, 1]
                arr = np.load(f)   
                arr = arr.astype(np.float32)
                if len(arr.shape) == 2:
                    arr = np.expand_dims(arr, axis=-1)
                # transpose to C*H*W
                arr = np.transpose(arr, (2, 0, 1))
                arr = arr[:in_channels] # ic*H*W
                # save to dict
                guided_arr_dict[str(file)[0:-4]] = arr
    return guided_arr_dict

def normalization(arr):
    return 2 * (arr + 5) / 45 - 1 

def split_guide_eval(arrs, guided_rate=0.6, in_channels = 1):
    guided_arrs = []
    eval_arrs = []
    for arr in arrs:
        # arr: HW
        mask = ~np.isnan(arr)  # 0 is nan
        mask_indices = np.argwhere(mask) # get the indices of non-nan elements
        selected_indices = mask_indices[np.random.choice(len(mask_indices), int(len(mask_indices) * (1-guided_rate)), replace=False)] # get the indices of eval part
        for idx in selected_indices:
            mask[tuple(idx)] = np.nan
        guided_arr = mask * arr  
        eval_mask = np.isnan(mask)
        eval_mask[eval_mask==0] = np.nan
        eval_arr =  eval_mask * arr #
        guided_arrs.append(guided_arr)
        eval_arrs.append(eval_arr)
    return np.array(guided_arrs), np.array(eval_arrs)


def split_guided_eval(arr, guided_rate):
    """
    Args:
        arrs: List of arrays (shape [H, W]) or a single array (shape [N, H, W]).
        guided_rate: Fraction of data to keep for guidance (e.g., 0.8 means 80% guided, 20% eval).
    Returns:
        guided_arrs: Arrays with only the guided part (non-guided set to NaN).
        eval_arrs: Arrays with only the eval part (non-eval set to NaN).
    """
    arr = np.asarray(arr)  # Ensure input is a NumPy array

    # Precompute masks for non-NaN positions
    non_nan_mask = ~np.isnan(arr)  # CHW
    num_non_nan = non_nan_mask.sum(axis=(1, 2))  # Number of non-NaN per array

    # Generate random indices for eval (non-guided) positions
    rng = np.random.default_rng()  # Random number generator
    eval_indices = []
    for i in range(len(arr)):
        mask_indices = np.argwhere(non_nan_mask[i])  # HW non-nan indices of i-th array
        n_eval = int(len(mask_indices) * (1 - guided_rate)) # num of eval
        selected = rng.choice(len(mask_indices), size=n_eval, replace=False) # index of mask_indices for eval 
        eval_indices.append(mask_indices[selected]) # CHW 

    # Create eval_mask (1=eval, 0=guided or originally NaN)
    eval_mask = np.zeros_like(arr, dtype=bool)
    for i, idx in enumerate(eval_indices):
        eval_mask[i, idx[:, 0], idx[:, 1]] = True

    # Generate guided_arrs and eval_arrs
    # eval_mask is T, others: nan
    eval_arrs = np.where(~eval_mask, np.nan, arr) 
    # guided_arrs: eval_mask T is nan or nan is nan
    guided_arrs = np.where(eval_mask | ~non_nan_mask, np.nan, arr)  # Keep guided + originally non-NaN

    return guided_arrs, eval_arrs  # Remove batch dim if input was single array


def split_guided_eval_batch_size(batchsize, arr, guided_rate):
    guided_arrs = []
    eval_arrs = []
    for _ in range(batchsize):
        guided_arr, eval_arr = split_guided_eval(arr, guided_rate)
        guided_arrs.append(guided_arr)
        eval_arrs.append(eval_arr)
    if torch_npu.npu.is_available():
        device = th.device("npu")
        print("Using Ascend")
    else:
        device = th.device("cpu")
        print("Using CPU")
    guided_y = th.from_numpy(np.array(guided_arrs)).float().to(f'npu:{device}' if isinstance(device, int) else device)
    eval_y = th.from_numpy(np.array(eval_arrs)).float().to(f'npu:{device}' if isinstance(device, int) else device)
    # eval_y = np.numpy(eval_arrs, dtype=np.float32)
    return guided_y, eval_y
        


def calculate_loss(pred_arr, gt_arr, loss="l1"):
    pred_arr = pred_arr.permute(0,3,1,2)   # B C H W
    mask = ~th.isnan(gt_arr) # B C H W  1,0

    # pred_arr = pred_arr[mask] 
    # gt_arr = gt_arr[mask]
   
    pred_arr = th.nan_to_num(pred_arr * mask, 0.0)
    gt_arr = th.nan_to_num(gt_arr * mask, 0.0)
    if loss == "l2" or loss == "mse":
        loss_values = F.mse_loss(pred_arr, gt_arr, reduction='none')  # (B, C, H, W)
    elif loss == "l1":
        loss_values = F.l1_loss(pred_arr, gt_arr, reduction='none')  # (B, C, H, W)
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    loss_per_channel = loss_values.mean(dim=(-2, -1)).mean(dim=0)  # (B, C)  -> C
    return loss_per_channel.to("cpu").numpy()  
    # .to("cpu").item()
