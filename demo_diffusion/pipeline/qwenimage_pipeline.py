from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Any, List, Optional, Union

import numpy as np
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import PreTrainedTokenizerBase

from demo_diffusion import path as path_module
from demo_diffusion.dynamic_import import import_from_diffusers
from demo_diffusion.model import (
    make_scheduler,
    make_tokenizer,
    unload_torch_model,
)
from demo_diffusion.pipeline.diffusion_pipeline import DiffusionPipeline
from demo_diffusion.pipeline.type import PIPELINE_TYPE

# Import models from diffusers
QwenImageTransformer2DModel = import_from_diffusers("QwenImageTransformer2DModel", "diffusers.models")
AutoencoderKLQwenImage = import_from_diffusers("AutoencoderKLQwenImage", "diffusers.models")

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class QwenImagePipeline(DiffusionPipeline):
    """
    Application showcasing the acceleration of Qwen-Image pipelines using Nvidia TensorRT.
    """

    def __init__(
        self,
        version="qwen-image",
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        guidance_scale: float = 4.0,
        true_cfg_scale: float = 4.0,
        max_sequence_length: int = 256,
        **kwargs,
    ):
        """
        Initializes the Qwen-Image pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be "qwen-image"
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline.
            guidance_scale (`float`, defaults to 4.0):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt.
            true_cfg_scale (`float`, defaults to 4.0):
                True CFG scale for Qwen-Image model.
            max_sequence_length (`int`, defaults to 256):
                Maximum sequence length to use with the `prompt`.
        """
        super().__init__(version=version, pipeline_type=pipeline_type, **kwargs)

        self.guidance_scale = guidance_scale
        self.true_cfg_scale = true_cfg_scale
        self.max_sequence_length = max_sequence_length

    @classmethod
    def FromArgs(cls, args: argparse.Namespace, pipeline_type: PIPELINE_TYPE) -> QwenImagePipeline:
        """Factory method to construct a `QwenImagePipeline` object from parsed arguments."""
        MAX_BATCH_SIZE = 4
        DEVICE = "cuda"
        DO_RETURN_LATENTS = False

        # Resolve all paths.
        dd_path = path_module.resolve_path(
            cls.get_model_names(pipeline_type), args, pipeline_type, cls._get_pipeline_uid(args.version)
        )

        return cls(
            dd_path=dd_path,
            version=args.version,
            pipeline_type=pipeline_type,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale if hasattr(args, 'true_cfg_scale') else 4.0,
            max_sequence_length=args.max_sequence_length if hasattr(args, 'max_sequence_length') else 256,
            bf16=args.bf16,
            low_vram=args.low_vram if hasattr(args, 'low_vram') else False,
            torch_fallback=args.torch_fallback if hasattr(args, 'torch_fallback') else None,
            weight_streaming=args.ws,
            max_batch_size=MAX_BATCH_SIZE,
            denoising_steps=args.denoising_steps,
            scheduler=args.scheduler,
            device=DEVICE,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            verbose=args.verbose,
            nvtx_profile=args.nvtx_profile,
            use_cuda_graph=args.use_cuda_graph,
            framework_model_dir=args.framework_model_dir,
            return_latents=DO_RETURN_LATENTS,
            torch_inference=args.torch_inference,
        )

    @classmethod
    def get_model_names(cls, pipeline_type: PIPELINE_TYPE, controlnet_type: str = None) -> List[str]:
        """Return a list of model names used by this pipeline."""
        return ["text_encoder", "tokenizer", "transformer", "vae", "scheduler"]

    @classmethod
    def _get_pipeline_uid(cls, version: str) -> str:
        """Return a unique identifier for the pipeline based on version."""
        return f"qwen-image-{version}"

    def is_native_export_supported(self, model_config: dict[str, Any]) -> bool:
        """Check if native export is supported for the given model configuration."""
        return True

    def _initialize_models(self, framework_model_dir, int8, fp8, fp4):
        """Initialize the Qwen-Image models."""
        # Load tokenizer
        self.tokenizer = make_tokenizer(
            self.version, self.pipeline_type, self.hf_token, framework_model_dir
        )

        # Set precision flags
        self.bf16 = True if int8 or fp8 or fp4 else self.bf16
        self.fp16 = True if not self.bf16 else False
        self.tf32 = True

        # Define models_args
        models_args = {
            "version": self.version,
            "pipeline": self.pipeline_type,
            "device": self.device,
            "hf_token": self.hf_token,
            "verbose": self.verbose,
            "framework_model_dir": framework_model_dir,
            "max_batch_size": self.max_batch_size,
        }

        # Initialize models based on stages
        if "text_encoder" in self.stages:
            from demo_diffusion.model.clip import CLIPModel
            self.models["text_encoder"] = CLIPModel(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                embedding_dim=3584,  # From config.json
                subfolder="text_encoder",
            )

        if "transformer" in self.stages:
            from demo_diffusion.model.diffusion_transformer import QwenImageTransformerModel
            self.models["transformer"] = QwenImageTransformerModel(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                subfolder="transformer",
                weight_streaming=self.weight_streaming,
            )

        if "vae" in self.stages:
            from demo_diffusion.model.vae import QwenImageVAEModel
            self.models["vae"] = QwenImageVAEModel(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                subfolder="vae",
            )

    def initialize_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        latent_timestep=None,
        image_latents=None,
        latents_dtype=torch.float32,
    ):
        """Initialize latents for diffusion process."""
        latents_shape = (batch_size, num_channels_latents, latent_height, latent_width)
        latents = torch.randn(
            latents_shape,
            device=self.device,
            dtype=latents_dtype,
            generator=self.generator,
        )
        return latents

    def encode_prompt(self, prompt, negative_prompt=""):
        """Encode text prompts using the text encoder."""
        self.profile_start("text_encoder", color="green")

        def tokenize(prompt, max_sequence_length):
            text_input_ids = (
                self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_overflowing_tokens=False,
                    return_length=False,
                    return_tensors="pt",
                )
                .input_ids.type(torch.int32)
                .to(self.device)
            )
            return text_input_ids

        text_input_ids = tokenize(prompt, self.max_sequence_length)

        if self.torch_inference or (self.torch_fallback and self.torch_fallback.get("text_encoder", False)):
            outputs = self.torch_models["text_encoder"](text_input_ids)
            text_encoder_output = outputs[0].clone()
        else:
            text_encoder_output = self.run_engine("text_encoder", {"input_ids": text_input_ids})["hidden_states"]

        self.profile_stop("text_encoder")
        return text_encoder_output

    def denoise_latents(self, latents, text_embeddings, timesteps):
        """Perform denoising using the transformer model."""
        self.profile_start("transformer", color="blue")

        # Prepare transformer inputs
        transformer_inputs = {
            "hidden_states": latents,
            "encoder_hidden_states": text_embeddings,
            "timestep": timesteps,
        }

        if self.torch_inference or (self.torch_fallback and self.torch_fallback.get("transformer", False)):
            noise_pred = self.torch_models["transformer"](**transformer_inputs)
        else:
            noise_pred = self.run_engine("transformer", transformer_inputs)["sample"]

        self.profile_stop("transformer")
        return noise_pred

    def decode_latents(self, latents):
        """Decode latents to images using the VAE."""
        self.profile_start("vae", color="red")

        if self.torch_inference or (self.torch_fallback and self.torch_fallback.get("vae", False)):
            images = self.torch_models["vae"].decode(latents)
        else:
            images = self.run_engine("vae", {"latent_sample": latents})["images"]

        self.profile_stop("vae")
        return images

    def run(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        height: int = 1328,
        width: int = 1328,
        batch_count: int = 1,
        num_warmup_runs: int = 5,
        use_cuda_graph: bool = False,
    ):
        """
        Run the Qwen-Image text-to-image generation pipeline.
        """
        # Ensure prompts are lists
        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt]

        batch_size = len(prompt)

        # Warmup
        if num_warmup_runs > 0:
            print("[I] Warming up...")
            for _ in range(num_warmup_runs):
                _ = self._generate_images(prompt, negative_prompt, height, width, warmup=True)

        # Generation
        all_images = []
        for i in range(batch_count):
            print(f"[I] Running Qwen-Image pipeline [{i+1}/{batch_count}]")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()

            images = self._generate_images(prompt, negative_prompt, height, width, warmup=False)
            all_images.append(images)

            if self.nvtx_profile:
                cudart.cudaProfilerStop()

        return all_images

    def _generate_images(self, prompt, negative_prompt, height, width, warmup=False):
        """Internal method to generate images."""
        batch_size = len(prompt)
        
        # Calculate latent dimensions
        latent_height = height // 8
        latent_width = width // 8
        num_channels_latents = 16  # From Qwen-Image config

        # Encode prompts
        text_embeddings = self.encode_prompt(prompt, negative_prompt)

        # Initialize latents
        latents = self.initialize_latents(
            batch_size, num_channels_latents, latent_height, latent_width
        )

        # Set up scheduler
        if not hasattr(self, 'scheduler_initialized'):
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                os.path.join(self.framework_model_dir, "scheduler")
            )
            self.scheduler_initialized = True

        self.scheduler.set_timesteps(self.denoising_steps)

        # Denoising loop
        for i, timestep in enumerate(self.scheduler.timesteps):
            # Prepare timestep
            timestep_tensor = torch.tensor([timestep] * batch_size, device=self.device)

            # Denoise
            noise_pred = self.denoise_latents(latents, text_embeddings, timestep_tensor)

            # Update latents
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        # Decode to images
        if not self.return_latents:
            images = self.decode_latents(latents)
        else:
            images = latents

        return images

    def save_images(self, prompt, images):
        """Save generated images to files."""
        # This would typically save images using PIL
        # For now, just print completion
        print(f"[I] Generated {len(images)} image(s) for prompts: {prompt}")