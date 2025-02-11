# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
# from download import find_model
# from models import DiT_models
# from custom_models import DiT_models
from models import VDT_models

import argparse

import numpy as np
from mmengine import Config
from mmengine.registry import MODELS
from copy import deepcopy
from utils_occworld.OccWorld.dataset import get_dataloader, get_nuScenes_label_name
from utils_occworld.OccWorld.utils.metric_util import MeanIoU, multi_step_MeanIou
from utils import load_checkpoint, custom_load_checkpoint

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    nframes_past = 5
    nframes = 11

    # Load model:
    # latent_size = args.image_size // 8
    # model = DiT_models[args.model](
    #     input_size=latent_size,
    #     num_classes=args.num_classes
    # ).to(device)

    # model = DiT_models[args.model](
    #     input_size=50,
    #     num_classes=args.num_classes,
    #     nframes=nframes,
    # ).to(device)

    # # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    # ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    # state_dict = find_model(ckpt_path)
    # model.load_state_dict(state_dict)
    # model.eval()  # important!


    model = VDT_models[args.model](
        depth=8,
        hidden_size=256,
        patch_size=2,
        num_heads=8,
        input_size=50,
        num_frames=nframes,
        in_channels=64,
        mode='video',
    )

    # model, _ = load_checkpoint(model, args.ckpt)
    model, _ = custom_load_checkpoint(model, args.ckpt)
    model = model.to(device)   
    model.eval()  # important!


    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # load config
    # cfg = Config.fromfile('utils/OccWorld/config/custom_train_occworld.py')
    cfg = Config.fromfile('utils_occworld/OccWorld/config/custom_occworld.py')
    import utils_occworld.OccWorld.model
    from utils_occworld.OccWorld.dataset import get_dataloader, get_nuScenes_label_name

    occ_vae = MODELS.build(cfg.model)
    occ_vae = occ_vae.to(device)
    occ_vae.init_weights()
    ckpt_occ_vae = torch.load(cfg.load_from, map_location='cpu')
    if 'state_dict' in ckpt_occ_vae:
        state_dict = ckpt_occ_vae['state_dict']
    else:
        state_dict = ckpt_occ_vae
    occ_vae.load_state_dict(state_dict, strict=True)

    occ_vae.eval()
    occ_vae.requires_grad_(False)

    label_name = get_nuScenes_label_name(cfg.label_mapping)
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(unique_label, cfg.get('ignore_label', -100), unique_label_str, 'sem', times=cfg.get('eval_length'))
    CalMeanIou_vox = multi_step_MeanIou([1], cfg.get('ignore_label', -100), ['occupied'], 'vox', times=cfg.get('eval_length'))

    CalMeanIou_sem.reset()
    CalMeanIou_vox.reset()

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
    )

    
    # all_latents_list, all_std_list = [], []
    # for i, batch_dict in enumerate(val_dataset_loader):
    #     input_occs, target_occs, metas = batch_dict
    #     output_dict = {}
    #     output_dict['target_occs'] = input_occs[:, nframes_past:nframes]

    #     input_occs = input_occs[:, :nframes].to(device)

    #     batch_size = input_occs.shape[0]
    #     cond_mask = torch.zeros([batch_size, nframes, 1, 1, 1]).to(device)
    #     cond_mask[:, :nframes_past] = 1
    #     cond_mask = cond_mask.view(-1, *cond_mask.shape[2:])

    #     with torch.no_grad():
    #         # Map input images to latent space + normalize latents:
    #         occ_z, occ_shapes = occ_vae.forward_encoder(input_occs)
    #         latents, occ_z_mu, occ_z_sigma, occ_logvar = occ_vae.sample_z(occ_z)
    #     # latents = latents.view(batch_size, nframes, *latents.shape[1:])
    #     all_latents_list.append(latents)
    #     std = latents.std().item()
    #     all_std_list.append(std)

    #     if i == 10:
    #         break

    # all_latents = torch.stack(all_latents_list).std().item() # 0.72  1.0 / 0.72 = 1.38889 

    sample_fn = model.forward

    with torch.no_grad():

        for batch_dict in val_dataset_loader:
            input_occs, target_occs, metas = batch_dict
            output_dict = {}
            output_dict['target_occs'] = input_occs[:, nframes_past:nframes]

            input_occs = input_occs[:, :nframes].to(device)

            batch_size = input_occs.shape[0]
            mask = torch.ones([batch_size, nframes, 1, 1]).to(device)
            mask[:, :nframes_past] = 0
            # cond_mask = cond_mask.view(-1, *cond_mask.shape[2:])

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                occ_z, occ_shapes = occ_vae.forward_encoder(input_occs)
                x, occ_z_mu, occ_z_sigma, occ_logvar = occ_vae.sample_z(occ_z)
                x = x.mul_(1.38889)
            x = x.view(batch_size, nframes, *x.shape[1:])

            # std = latents.std().item()

            z = torch.randn(*x.shape, device=device)
            # cond_frame = latents
            # model_kwargs = dict(nframes=nframes, cond_frame=cond_frame, cond_mask=cond_mask)

            # Sample images:
            # samples = diffusion.p_sample_loop(
            #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            # )
            
            z = z.permute(0, 2, 1, 3, 4)
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, progress=True, device=device,
                raw_x=x, mask=mask
            )

            # samples = cond_frame

            samples = samples.permute(1, 0, 2, 3, 4) * mask + x.permute(2, 0, 1, 3, 4) * (1-mask)
            samples = samples.permute(1, 2, 0, 3, 4)

            # frames = samples.view(batch_size, nframes, *samples.shape[1:])
            frames_pred = samples[:, nframes_past:]
            frames_pred = frames_pred.contiguous().view(-1, *frames_pred.shape[2:])        

            # z_q_predict = occ_vae.forward_decoder(frames_pred, occ_shapes, output_dict['target_occs'].shape)
            z_q_predict = occ_vae.forward_decoder(frames_pred / 1.38889, occ_shapes, output_dict['target_occs'].shape)
            output_dict['logits'] = z_q_predict
            pred = z_q_predict.argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)
            
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou

            result_dict = output_dict

            if result_dict.get('target_occs', None) is not None:
                target_occs = result_dict['target_occs']
            target_occs_iou = deepcopy(target_occs)
            target_occs_iou[target_occs_iou != 17] = 1
            target_occs_iou[target_occs_iou == 17] = 0

            CalMeanIou_sem._after_step(result_dict['sem_pred'], target_occs)
            CalMeanIou_vox._after_step(result_dict['iou_pred'], target_occs_iou)

            val_miou, _ = CalMeanIou_sem._after_epoch()
            val_iou, _ = CalMeanIou_vox._after_epoch()

            print()

    # # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # # Create sampling noise:
    # n = len(class_labels)
    # z = torch.randn(n, 4, latent_size, latent_size, device=device)
    # y = torch.tensor(class_labels, device=device)

    # # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    # # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    # print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-S/2")

    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
