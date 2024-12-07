#init model
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from models.unet import UNet2DConditionModel
from models.vae import AutoencoderKL
import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,)

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from copy import deepcopy
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import torch.nn.functional as F


from dataset_deepfuniture_img2img import Deepfurniture_Dataset_V1, collate_fn, collate_fn_embedding
from embedding_data import EmbeddingData
from utils import load_safetensors, get_dtype_training
from args import parse_args
if is_wandb_available():
    import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from models.attention import skip_encoder_hidden_state, SkipAttnProcessor
from trainer import LitSDPix2PixControlnext, TrainableParameters, LitSDCATDF_v1
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0.dev0")

logger = get_logger(__name__)


class ModelTrainManager:
    def __init__(self, args):
        self.args = args
        self.unet = None
        self.controlnext = None
        self.vae = None
        self.noise_scheduler = None
        self.tokenizer = None
        self.text_encoder = None

    def init_unet(self):
        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                self.args.unet_model_name_or_path, revision=self.args.revision, variant=self.args.variant
            )
            skip_encoder_hidden_state(self.unet,cross_attn_cls= SkipAttnProcessor )
            logger.info("Successfully initialized UNet model")
        except Exception as e:
            logger.error(f"Failed to initialize UNet model: {e}")


    def init_vae(self):
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.args.revision,
                variant=self.args.variant,
            )
            logger.info("Successfully initialized VAE model")
        except Exception as e:
            logger.error(f"Failed to initialize VAE model: {e}")

    def init_tokenizer(self): 
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer", revision=self.args.revision, use_fast=False
            )
            logger.info("Successfully initialized tokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")

    def init_noise_scheduler(self):
        try:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type = self.args.prediction_type
            )
            logger.info("Successfully initialized noise scheduler")
        except Exception as e:
            logger.error(f"Failed to initialize noise scheduler: {e}")

    def init_text_encoder(self):
        try:
            text_encoder_config = PretrainedConfig.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=self.args.revision,
            )
            model_class = text_encoder_config.architectures[0]

            if model_class == "CLIPTextModel":
                from transformers import CLIPTextModel
                self.text_encoder = CLIPTextModel.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    revision=self.args.revision,
                )
                logger.info("Successfully initialized text encoder: CLIPTextModel")
            else:
                raise ValueError(f"{model_class} is not supported.")
        except Exception as e:
            logger.error(f"Failed to initialize text encoder: {e}")
    
    def run_load_model(self,is_data_embedding):
        if is_data_embedding == True:
            self.init_vae()
            self.init_tokenizer()
            self.init_text_encoder()

            return self.vae, self.tokenizer,self.text_encoder
        
        else:
            self.init_unet()
            self.init_noise_scheduler()
            
            return self.unet, self.noise_scheduler


def setup_accelerator(args):
    """Setup Accelerator and initialize wandb tracker."""
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=args.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    # Initialize wandb tracker
    accelerator.init_trackers(
        project_name="train_cat_inpainting",
        config={
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "num_train_epochs": args.num_train_epochs,
            "resolution": args.resolution,
        },
    )
    return accelerator

def create_dataloader(args):
    dtype = get_dtype_training(args.mixed_precision)
    dataset = Deepfurniture_Dataset_V1(
        args.path_to_save_data_embedding,
        input_type="embedding",
        dtype=dtype,
        image_size=args.resolution
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
    )
    return dataloader


def main():
    args = parse_args()
    accelerator = setup_accelerator(args)
    
    # Load models, data, and optimizer
    model_trainable = ModelTrainManager(args)
    
    # Embedding data generation (if specified)
    if args.save_embeddings_to_npz == True:
        dataset = Deepfurniture_Dataset_V1(
            args.dataset_path,
            input_type=args.input_type,
            dtype=args.mixed_precision,
            image_size=args.resolution
        )
        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_fn_embedding,
        )
        
        vae, tokenizer, text_encoder = model_trainable.run_load_model(is_data_embedding=True)
        embedding = EmbeddingData(
            train_loader, vae, tokenizer, text_encoder,
            args.mixed_precision, args.path_to_save_data_embedding, args
        )
        
        del vae
        del tokenizer
        del text_encoder
        embedding.save_embeddings_to_npz('cuda')
    
    unet, scheduler = model_trainable.run_load_model(is_data_embedding=False)

    del model_trainable
    # print(scheduler.config.prediction_type)

    
    train_dataloader = create_dataloader(args)
    
    wandb_logger = WandbLogger(
        project="train cat inpainting",
        log_model=False)
    
    #Initialize the trainable params config
    param_train_config = TrainableParameters(unet= unet, learning_rate= args.learning_rate)
    
     # Initialize the Lightning Model
    lit_model = LitSDCATDF_v1(
        unet=unet,
        noise_scheduler=scheduler,    
        device_train='cuda',
        scheduler_optimizer_config=param_train_config,
    )

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="model-{epoch:02d}-{train_loss:.4f}",
        save_top_k=5,
        mode="min",
    )

     # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer configuration
    trainer = Trainer(
        max_epochs=args.num_train_epochs,         
        accelerator= 'gpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=1,
        precision=16 if args.mixed_precision == "fp16" else 32,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    # Train the model
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader
    )

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "final_checkpoint.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)

    print(f"Training complete")

main()