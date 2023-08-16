import torch
import comfy.model_management
import math
import numpy as np
import latent_preview
from comfy.sample import *

from . import samplers_advanced

def sample_refined(model, refiner_model, noise, steps, cfg, sampler_name, scheduler, positive, negative, refiner_positive, refiner_negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    """
    from comfy.sample.sample
    """
    device = comfy.model_management.get_torch_device()

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    real_model = None
    comfy.model_management.load_model_gpu(model)
    real_model = model.model

    real_refiner_model = None
    comfy.model_management.load_model_gpu(refiner_model)
    real_refiner_model = refiner_model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)
    refiner_positive_copy = broadcast_cond(refiner_positive, noise.shape[0], device)
    refiner_negative_copy = broadcast_cond(refiner_negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())
    refiner_models = load_additional_models(positive, negative, refiner_model.model_dtype())

    sampler = samplers_advanced.KSamplerWithRefiner(real_model, real_refiner_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, refiner_positive_copy, refiner_negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback_function=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.cpu()

    cleanup_additional_models(models)
    cleanup_additional_models(refiner_model)
    return samples


def ksampler_refined(model, refiner_model, seed, steps, cfg, sampler_name, scheduler, positive, negative, refiner_positive, refiner_negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    """
    from nodes.common_ksampler
    """
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = sample_refined(model, refiner_model, noise, steps, cfg, sampler_name, scheduler, positive, negative, refiner_positive, refiner_negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

class KSamplerWithRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "refiner_model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "refiner_positive": ("CONDITIONING", ),
                    "refiner_negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, refiner_model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, refiner_positive, refiner_negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return ksampler_refined(model, refiner_model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, refiner_positive, refiner_negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
