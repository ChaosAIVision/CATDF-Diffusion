from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import islice

class EmbeddingData():
    def __init__(self,data_loader, vae, tokenizer,text_encoder, weight_dtype, base_dir,args ):

        self.data_loader = data_loader
       
        if weight_dtype == 'bf16':
            self.weight_dtype = torch.bfloat16
        else:
            weight_dtype == torch.float32
        
        self.vae = vae.to(dtype= self.weight_dtype, device = 'cuda')
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(dtype= self.weight_dtype, device = 'cuda')
        self.base_dir = base_dir
        self.args = args
        if args.tensor_dtype_save == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def save_embeddings_to_npz(self, device):
        npz_data = {} 
        metadata_records = [] 
        from itertools import islice

        for i, batch in tqdm(enumerate(islice(self.data_loader, 1)), total= 1, desc="Processing batches"): #test with data loader 1 items
        # for i, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader), desc="Processing batches"):
            #encoder latent and   
            latents_target = self.vae.encode(batch["latents_target"].to(dtype=self.weight_dtype,device = 'cuda')).latent_dist.sample()
            latents_target = latents_target * self.vae.config.scaling_factor
            
            latents_masked = self.vae.encode(batch["latents_masked"].to(dtype=self.weight_dtype,device = 'cuda')).latent_dist.sample()
            latents_masked = latents_masked * self.vae.config.scaling_factor
            
            #encoder text
            input_text = (self.tokenizer(' fill image and make picture high quality and harmonization ', max_length=self.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids).to('cuda')
            encoder_hidden_state = self.text_encoder(input_text, return_dict=False)[0]  

            # Get image conditional     
            mask_pixel_values = batch["mask_pixel_values"].to(dtype=self.weight_dtype,device = 'cuda')
            fill_pixel_values = self.vae.encode(batch["fill_pixel_values"].to(dtype=self.weight_dtype,device = 'cuda')).latent_dist.sample()
            fill_pixel_values = fill_pixel_values * self.vae.config.scaling_factor



            #build meta data
            npz_data[f"latents_target_{i}"] = latents_target.to(self.dtype).detach().cpu().numpy()
            npz_data[f"latents_masked_{i}"] = latents_masked.to(self.dtype).detach().cpu().numpy()
            npz_data[f"mask_pixel_values_{i}"] = mask_pixel_values.to(self.dtype).detach().cpu().numpy()
            npz_data[f"encoder_hidden_state_{i}"] = encoder_hidden_state.to(self.dtype).detach().cpu().numpy()
            npz_data[f"fill_pixel_values_{i}"] = fill_pixel_values.to(self.dtype).detach().cpu().numpy()




            metadata_records.append({
                "batch_index": i,
                "latents_target_key": f"latents_target_{i}",
                "latents_masked__key": f"latents_masked_{i}",
                "mask_pixel_values__key": f"mask_pixel_values_{i}",
                "encoder_hidden_state_key": f"encoder_hidden_state_{i}",
                "fill_pixel_values_key": f"fill_pixel_values_{i}"
            })

        np.savez_compressed(os.path.join(self.base_dir, "embeddings_data.npz"), **npz_data)
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(os.path.join(self.base_dir, "metadata.parquet"))
            
