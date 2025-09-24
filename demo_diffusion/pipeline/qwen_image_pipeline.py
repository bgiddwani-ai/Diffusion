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
    CLIPModel,
    make_scheduler,
    make_tokenizer,
    unload_torch_model,
    VAEModel,
)
from demo_diffusion.pipeline.diffusion_pipeline import DiffusionPipeline
from demo_diffusion.pipeline.type import PIPELINE_TYPE

# Import models from diffusers
QwenImageTransformer2DModel = import_from_diffusers("QwenImageTransformer2DModel", "diffusers.models")
AutoencoderKLQwenImage = import_from_diffusers("AutoencoderKLQwenImage", "diffusers.models")

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class QwenImageTransformerModel:
    """Qwen-Image Transformer model for TensorRT optimization."""
    
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=4,
        fp16=False,
        tf32=False,
        bf16=False,
        subfolder="transformer",
        **kwargs,
    ):
        self.version = version
        self.device = device
        self.hf_token = hf_token
        self.verbose = verbose
        self.framework_model_dir = framework_model_dir
        self.max_batch_size = max_batch_size
        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16
        self.subfolder = subfolder
        
    def get_model(self, torch_inference=""):
        """Load the Qwen-Image transformer model."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_path = os.path.join(self.framework_model_dir, self.subfolder)
        
        dtype = torch.bfloat16 if self.bf16 else torch.float16 if self.fp16 else torch.float32
        model = QwenImageTransformer2DModel.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(self.device).eval()
        
        return model


class QwenImageVAEModel:
    """Qwen-Image VAE model for TensorRT optimization."""
    
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=4,
        fp16=False,
        tf32=False,
        bf16=False,
        subfolder="vae",
        **kwargs,
    ):
        self.version = version
        self.device = device
        self.hf_token = hf_token
        self.verbose = verbose
        self.framework_model_dir = framework_model_dir
        self.max_batch_size = max_batch_size
        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16
        self.subfolder = subfolder
        
    def get_model(self, torch_inference=""):
        """Load the Qwen-Image VAE model."""
        model_path = os.path.join(self.framework_model_dir, self.subfolder)
        
        dtype = torch.bfloat16 if self.bf16 else torch.float16 if self.fp16 else torch.float32
        model = AutoencoderKLQwenImage.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(self.device).eval()
        
        return model


class QwenImageTextEncoderModel:
    """Qwen-Image Text Encoder model for TensorRT optimization."""
    
    def __init__(
        self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=4,
        fp16=False,
        tf32=False,
        bf16=False,
        subfolder="text_encoder",
        **kwargs,
    ):
        self.version = version
        self.device = device
        self.hf_token = hf_token
        self.verbose = verbose
        self.framework_model_dir = framework_model_dir
        self.max_batch_size = max_batch_size
        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16
        self.subfolder = subfolder
        
    def get_model(self, torch_inference=""):
        """Load the Qwen-Image text encoder model."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_path = os.path.join(self.framework_model_dir, self.subfolder)
        
        dtype = torch.bfloat16 if self.bf16 else torch.float16 if self.fp16 else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(self.device).eval()
        
        return model


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
        self.stages = ["text_encoder", "transformer", "vae"]
        
        # Model configuration
        self.latent_channels = 16  # From Qwen-Image config
        self.vae_scale_factor = 8  # Standard VAE scale factor

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
            true_cfg_scale=getattr(args, 'true_cfg_scale', 4.0),
            max_sequence_length=getattr(args, 'max_sequence_length', 256),
            bf16=args.bf16,
            low_vram=getattr(args, 'low_vram', False),
            torch_fallback=getattr(args, 'torch_fallback', None),
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
        return ["text_encoder", "transformer", "vae"]

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
        from transformers import Qwen2Tokenizer
        tokenizer_path = os.path.join(framework_model_dir, "tokenizer")
        self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)

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
            self.models["text_encoder"] = QwenImageTextEncoderModel(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                subfolder="text_encoder",
            )

        if "transformer" in self.stages:
            self.models["transformer"] = QwenImageTransformerModel(
                **models_args,
                fp16=self.fp16,
                tf32=self.tf32,
                bf16=self.bf16,
                subfolder="transformer",
            )

        if "vae" in self.stages:
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

        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        # Encode text
        if self.torch_inference or (self.torch_fallback and self.torch_fallback.get("text_encoder", False)):
            with torch.no_grad():
                text_encoder_output = self.torch_models["text_encoder"](
                    input_ids=text_input_ids
                ).last_hidden_state
        else:
            text_encoder_output = self.run_engine("text_encoder", {"input_ids": text_input_ids})["hidden_states"]

        self.profile_stop("text_encoder")
        return text_encoder_output

    def denoise_latents(self, latents, text_embeddings, timestep):
        """Perform denoising using the transformer model."""
        self.profile_start("transformer", color="blue")

        # Prepare transformer inputs
        transformer_inputs = {
            "hidden_states": latents,
            "encoder_hidden_states": text_embeddings,
            "timestep": timestep,
        }

        if self.torch_inference or (self.torch_fallback and self.torch_fallback.get("transformer", False)):
            with torch.no_grad():
                noise_pred = self.torch_models["transformer"](**transformer_inputs).sample
        else:
            noise_pred = self.run_engine("transformer", transformer_inputs)["sample"]

        self.profile_stop("transformer")
        return noise_pred

    def decode_latents(self, latents):
        """Decode latents to images using the VAE."""
        self.profile_start("vae", color="red")

        if self.torch_inference or (self.torch_fallback and self.torch_fallback.get("vae", False)):
            with torch.no_grad():
                images = self.torch_models["vae"].decode(latents).sample
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
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # Encode prompts
        text_embeddings = self.encode_prompt(prompt, negative_prompt)

        # Initialize latents
        latents = self.initialize_latents(
            batch_size, self.latent_channels, latent_height, latent_width
        )

        # Set up scheduler
        if not hasattr(self, 'scheduler_initialized'):
            scheduler_path = os.path.join(self.framework_model_dir, "scheduler")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
            self.scheduler_initialized = True

        self.scheduler.set_timesteps(self.denoising_steps)

        # Denoising loop
        for i, timestep in enumerate(self.scheduler.timesteps):
            # Prepare timestep
            timestep_tensor = torch.tensor([timestep] * batch_size, device=self.device, dtype=torch.float32)

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
        from PIL import Image
        import os
        
        if not isinstance(images, list):
            images = [images]
            
        for i, image_tensor in enumerate(images):
            if isinstance(image_tensor, torch.Tensor):
                # Convert tensor to PIL Image
                image_np = image_tensor.cpu().numpy()
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
                
                if len(image_np.shape) == 4:  # Batch dimension
                    for j, img in enumerate(image_np):
                        img = img.transpose(1, 2, 0)  # CHW -> HWC
                        pil_image = Image.fromarray(img)
                        filename = f"qwen_image_output_{i}_{j}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        pil_image.save(filepath)
                        print(f"[I] Saved image: {filepath}")
                else:
                    image_np = image_np.transpose(1, 2, 0)  # CHW -> HWC
                    pil_image = Image.fromarray(image_np)
                    filename = f"qwen_image_output_{i}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    pil_image.save(filepath)
                    print(f"[I] Saved image: {filepath}")
        
        print(f"[I] Generated {len(images)} image(s) for prompts: {prompt}")

    def teardown(self):
        """Clean up resources."""
        for model_name in self.stages:
            if model_name in self.models:
                unload_torch_model(self.models[model_name])
        
        # Clean up engines if they exist
        if hasattr(self, 'engine'):
            for engine_name in self.engine:
                if self.engine[engine_name]:
                    del self.engine[engine_name]
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()