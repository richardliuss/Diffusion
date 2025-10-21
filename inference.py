"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import time

import numpy as np
import torch_npu
import torch as th
import blobfile as bf
import torch.distributed as dist
import torch.nn.functional as F

import xarray as xr

import dist_util 
import logger 

from script_util_v2 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    # classifier_defaults,  # Added
    create_model_and_diffusion,
    # create_classifier, # Added
    add_dict_to_argparser,
    args_to_dict,
)
from guided_util import * 

def get_gaussian_kernel(size, sigma):
    coords = th.arange(size, dtype=th.float32)
    coords -= size // 2
    g = th.exp(-(coords**2) / (2 * sigma**2))  
    g = g.outer(g)  
    # return (g / g.sum()).view(1, 1, size, size)  
    return g.view(1, 1, size, size)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='./')  # 配置日志输出目录

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())  # Unet Model 和 Gaussian Diffusion
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(f'npu:{dist_util.dev()}' if isinstance(dist_util.dev(), int) else dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  

    logger.log("Loading sparse data for guiding...")
    guided_arr_dict = {}
    for file in bf.listdir(args.sparse_data_path):
        # file only contains the name of the file, not the full path
        # print(file[-11:-9])
        if file.endswith(".nc"):
            print(file)
            path = os.path.join(args.sparse_data_path,file)
            ds = xr.open_dataset(path)
            arr = ds['temperature'].values
            arr = arr.astype(np.float32)  
            if len(arr.shape) == 4:
                arr = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[3])
                
            guided_arr_dict[str(file)[0:-3]] = arr  # .npy->.nc
    print(guided_arr_dict.keys()) 
    logger.log("Successfully load the guided data!")

    if torch_npu.npu.is_available():
        device = th.device("npu")
        print("Using Ascend")
    else:
        device = th.device("cpu")
        print("Using CPU")
    softmask_kernel = get_gaussian_kernel(5, 1.0).float().to(f'npu:{device}' if isinstance(device, int) else device) 

    print(softmask_kernel)

    def cond_fn(x, t, p_mean_var, **model_kwargs):
        y = model_kwargs.get("y")
        assert y is not None
        x = p_mean_var['pred_xstart'] # x0
        s = args.grad_scale
        gradient = 2 * ( y - x )   
        gradient = th.nan_to_num(gradient, nan=0.0)  
        if args.use_softmask:
            size = softmask_kernel.shape[-1]
            gradient = F.conv2d(
            gradient, 
            softmask_kernel.expand(gradient.shape[1], 1, size, size), 
            stride=1,
            padding= size // 2,
            groups=gradient.shape[1] 
            )
        return gradient * s
    
    def cond_fn_3d(x, t, p_mean_var, **model_kwargs):
        assert y is not None
        x = p_mean_var['pred_xstart']  # x0
        s = args.grad_scale
        gradient = 2 * (y - x) 
        gradient = th.nan_to_num(gradient, nan=0.0) 

        if args.use_softmask:
            size = softmask_kernel.shape[-1]
            gradient = gradient.unsqueeze(1)  
            gradient = F.conv3d(
                gradient, 
                softmask_kernel,  
                stride=1,
                padding=size // 2,
                groups=1 
            )  # B1CHW 
            gradient = gradient.squeeze(1)
        return gradient * s    
    

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # outputdir
    date = time.strftime("%m%d")
    if args.dynamic_guided:
        config = f"dyn_next={args.dynamic_guided_with_next}_r={args.guided_rate}_sigma={args.use_sigma}"
    else:
        config = f"s={args.grad_scale}_r={args.guided_rate}_loss={args.loss_model}_softmask={args.use_softmask}_sigma={args.use_sigma}"

    out_dir = os.path.dirname(os.environ.get("DIFFUSION_SAMPLE_LOGDIR", logger.get_dir()))
    out_dir=os.path.join(out_dir, date, config)
    os.makedirs(out_dir, exist_ok=True)


    logger.log("sampling...")
    
    loss_preds = []
    loss_guideds = []
    losses = []
    for key, guided_arr in guided_arr_dict.items():
        all_images = []
        all_loss_pred = []
        all_loss_guided = []
        all_loss = []
        logger.log(f"sampling {key}...")
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}  
            guided_y, eval_y = split_guided_eval_batch_size(args.batch_size, guided_arr, args.guided_rate) 
            model_kwargs["y"] = normalization(guided_y) 
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )  
            sample = sample_fn(
                model_fn,  
                (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),  # 形状，第二个已修改
                clip_denoised=args.clip_denoised,  
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,  
                use_sigma = args.use_sigma,  
                dynamic_guided=args.dynamic_guided,  
                dynamic_guided_with_next=args.dynamic_guided_with_next,  
                device=dist_util.dev(),
            )  


            sample = ((sample + 1) * 22.5 - 5 ).clamp(-5, 40)  
            sample = sample.permute(0, 2, 3, 1)  
            sample = sample.contiguous() 

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            
            loss_pred = calculate_loss(sample, eval_y, args.loss_model)
            loss_guided = calculate_loss(sample, guided_y, args.loss_model)
            loss_total = (1-args.guided_rate)*loss_pred + args.guided_rate * loss_guided
            all_loss_pred.append(loss_pred)
            all_loss_guided.append(loss_guided)
            all_loss.append(loss_total)
            
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            logger.log(f"created {len(all_images) * args.batch_size} samples")
       
        arr = np.concatenate(all_images, axis=0)
        pred_loss = np.mean(all_loss_pred, axis=0)
        guided_loss = np.mean(all_loss_guided, axis=0)
        total_loss = np.mean(all_loss, axis=0)
        loss_preds.append(pred_loss)
        loss_guideds.append(guided_loss)
        losses.append(total_loss)
        
        if dist.get_rank() == 0:
            logger.log(f"Loss of {key} pred_loss: {pred_loss}, guided_loss: {guided_loss}, total_loss: {total_loss}")
            shape_str = "x".join([str(x) for x in arr.shape]) 
            # outputpath
            out_path = os.path.join(out_dir, f"{key}_sample{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)
            logger.log(f"sampling {key} complete")
    
    dist.barrier() 
    logger.log("Complete All Sample!")
    logger.log(f"Compute Total Loss: pred_loss: {np.mean(loss_preds, axis=0)}, guided_loss: {np.mean(loss_guideds, axis=0)}, total_loss: {np.mean(losses, axis=0)}")


def create_argparser():
    defaults = dict(
        image_size_H = 180,
        image_size_W = 360,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        use_sigma=True,
        model_path="./model.pt",
        sparse_data_path="./sparse_data",
        grad_scale=1.0,  # when 0: sample from the base diffusion model
        use_softmask=False,
        dynamic_guided = False,
        dynamic_guided_with_next = False,
        guided_rate=0.6,
        loss_model="l1",
        use_fp16=False,
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()

    # if dist.is_initialized():
    #     dist.destroy_process_group()
