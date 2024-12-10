from lightning.pytorch import LightningModule
import torch
from diffusers.training_utils import EMAModel, compute_snr
import torch.nn.functional as F
from typing import Iterable, Optional, Any
from torchmetrics import MeanMetric
from copy import deepcopy
from utils import get_dtype_training, log_data_make_loss_high , compute_dream_and_update_latents_for_inpaint
import os
from models.unet import UNet2DConditionModel
from models.attention import skip_encoder_hidden_state
from models.utils_model import get_trainable_module
from dataclasses import dataclass

@dataclass
class TrainableParameters:

    unet:UNet2DConditionModel
    learning_rate: float

    def get_unet_trainable_parameters(self):
        
        trainable_params = get_trainable_module(self.unet,"attention")        
        return trainable_params

    def get_optimizer_parameters(self):
        
        optimizer_params = []
        # optimizer_params.append({"params": self.unet.parameters(),"lr": self.learning_rate})

        unet_trainable_params = self.get_unet_trainable_parameters()
        # for name, param in unet_trainable_params.items():
        #     optimizer_params.append({"params": param, "lr": self.learning_rate})
        seen_params = set()
        if isinstance(unet_trainable_params, torch.nn.ModuleList):
            for module in unet_trainable_params:
                for param in module.parameters():
                    if param not in seen_params:  # Kiểm tra tham số đã tồn tại chưa
                        optimizer_params.append({"params": param, "lr": self.learning_rate})
                        seen_params.add(param)  # Đánh dấu tham số đã thêm
        del seen_params

        return optimizer_params
        


    
class LitSDCATDF_v1(LightningModule):
    def __init__(self,
        unet,
        device_train,
        noise_scheduler,
        scheduler_optimizer_config,
        args
    ):
        
        super(LitSDCATDF_v1, self).__init__()
        self.save_hyperparameters()
        self.unet = unet
        self.device_train = device_train
        self.noise_scheduler= noise_scheduler
        self.scheduler_optimizer_config =scheduler_optimizer_config
        self.args = args

    def forward(self, 
                noisy_latent_input,
                time_steps,
                encoder_hidden_states,
                ):

        model_pred = self.unet(
            noisy_latent_input,
            time_steps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]

        return model_pred
    
    def training_step(self, batch, batch_idx):

        # Define latent_concat dim
        concat_dim = -2 
        
        # Get data_embedding
        device= self.device_train
        latents_target = batch["latents_target"].to(device)
        latent_masked = batch['latents_masked'].to(device)
        encoder_hidden_states = batch["encoder_hidden_state"].to(device)
        mask_pixel_values = batch['mask_pixel_values'].to(device)
        fill_pixel_values = batch['fill_pixel_values'].to(device)

        # # Create index to handle loss
        # index_fill_image = (batch['index_fill_image'].to('cpu'))
        # index_fill_image_list = index_fill_image.flatten().tolist()  
        # index_fill_image_list = [int(idx) for idx in index_fill_image_list] 

        # Log index make handle loss
        # log_complexity_data_idx_path = os.path.join(self.args.output_dir,'/log_complexity_data_idx.json')

        # Create timesteps
        bsz = latents_target.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device= device)

        # Concatenate conditional
        masked_latent_concat = torch.cat([latent_masked, fill_pixel_values], dim=concat_dim)

        height_mask, width_mask = mask_pixel_values.shape[2], mask_pixel_values.shape[3]        
        mask_latent=  torch.nn.functional.interpolate(mask_pixel_values, size=( height_mask// 8, width_mask // 8))
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        # Padding latent target to fit shape masked_latent and mask
        latent_target_concat = torch.cat([latents_target, torch.zeros_like(latents_target)], dim=concat_dim)


        # Add noise to latents_target
        noise = torch.randn_like(latent_target_concat)
        noisy_latents_target = self.noise_scheduler.add_noise(latent_target_concat, noise, timesteps)
  
        inpainting_latent_model_input = torch.cat([noisy_latents_target, mask_latent_concat, masked_latent_concat], dim=1)

    # DREAM integration
        noisy_latents_target, target = compute_dream_and_update_latents_for_inpaint(
            unet=self.unet,
            noise_scheduler=self.noise_scheduler,
            timesteps=timesteps,
            noise=noise,
            noisy_latents=noisy_latents_target,
            target=latent_target_concat,
            encoder_hidden_states=None,
            dream_detail_preservation=1.0,  # You can adjust this value
        )

        # Forward computation
        model_pred = self(
            noisy_latent_input= inpainting_latent_model_input,
            time_steps= timesteps,
            encoder_hidden_states = None,
        )

        # noise = noise.split(noise.shape[concat_dim] // 2, dim=concat_dim)[0]
        # # Remove padding of prediction to caculate loss         
        # model_pred = model_pred.split(model_pred.shape[concat_dim] // 2, dim=concat_dim)[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latent_target_concat, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = latent_target_concat
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

     
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(self.noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, 5.0 * torch.ones_like(timesteps)], dim=1).min(
            dim=1
        )[0]
        if self.noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()
        # loss_value = loss.item()

    #     log_data_make_loss_high(
    #     loss_value=loss_value,
    #     idx_list=index_fill_image_list,  
    #     loss_threshold=0.05,
    #     save_log_path=log_complexity_data_idx_path
    # )

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True,on_epoch=True,logger=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer= torch.optim.AdamW(
            self.scheduler_optimizer_config.get_optimizer_parameters(),
            lr = self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay= self.args.adam_weight_decay,
            eps= self.args.adam_epsilon
            
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200,eta_min=0
    )
        scheduler_optimizer_config ={
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': None
        }
        return [{"optimizer":optimizer,"scheduler": scheduler_optimizer_config, "interval": "step"}]