from pipeline.pipeline_stable_diffusion_cat import StableDiffusionInpaintConCat
from models.unet import UNet2DConditionModel
from utils import load_safetensors
from PIL import Image
import os
from models.attention import skip_encoder_hidden_state, SkipAttnProcessor
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,DDIMScheduler)
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

unet = UNet2DConditionModel.from_pretrained('/home/tiennv/trang/chaos/trash/output')
print('haha')
# load_safetensors(unet, r'/home/tiennv/chaos/unet.bin')
skip_encoder_hidden_state(unet,cross_attn_cls= SkipAttnProcessor )

# output_type = 'latent'
scheduler = DDIMScheduler.from_pretrained(
                'botp/stable-diffusion-v1-5-inpainting', subfolder="scheduler", prediction_type = 'epsilon')
generator = torch.Generator("cuda").manual_seed(0)
pipeline = StableDiffusionInpaintConCat.from_pretrained(
    'botp/stable-diffusion-v1-5-inpainting', 
    unet= unet, 
    dtype = torch.float32,
    # output_type = output_type,
    scheduler= scheduler,
    # generator= generator
).to('cuda')
promt = ''

image = Image.open('/home/tiennv/trang/chaos/trash/pipeline/idx_5/latents_masked.png')
condition_image =  Image.open('/home/tiennv/trang/chaos/trash/pipeline/idx_5/fill_pixel_values.png').convert("RGB")
condition_image= condition_image.resize((512,512))

mask_image = Image.open('/home/tiennv/trang/chaos/trash/pipeline/idx_5/mask_pixel_values.png').convert("RGB")
mask_image= mask_image.resize((512,512))


image = pipeline(
    prompt= promt, 
    image =image,fill_image =condition_image,
    mask_image=mask_image, return_dict = False,
    padding_mask_crop= None,
)[0]
((image[0].save(f'/home/tiennv/trang/chaos/trash/pipeline/idx_5/out.jpg')))