from pipeline.pipeline_stable_diffusion_inpainting_controlnext import StableDiffusionInpaintControlnextV1
from models.unet import UNet2DConditionModelControlNext, UNet2DConditionModelControlNext_XL
from models.controlnext import ControlNeXtModel, ControlNeXtModelXL, ControlNeXtModelSelfAttention
import torch
from utils import load_safetensors
from PIL import Image
import os
from models.attention import skip_encoder_hidden_state, SkipAttnProcessor
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

controlnext = ControlNeXtModelXL()
load_safetensors(controlnext, '/home/tiennv/trang/chaos/controlnext/khongquantrong/save_weight/controlnext.bin')
unet = UNet2DConditionModelControlNext_XL.from_pretrained('/home/tiennv/trang/chaos/controlnext/weight_pretrain/unet_inpainting')
load_safetensors(unet, r'/home/tiennv/trang/chaos/controlnext/khongquantrong/save_weight/diffusion_pytorch_model.bin',strict= False)
skip_encoder_hidden_state(unet,cross_attn_cls= SkipAttnProcessor )

output_type = 'latent'
scheduler = DDPMScheduler.from_pretrained(
                'botp/stable-diffusion-v1-5-inpainting', subfolder="scheduler", prediction_type = 'v_prediction')
pipeline = StableDiffusionInpaintControlnextV1.from_pretrained(
    'botp/stable-diffusion-v1-5-inpainting', 
    unet= unet, 
    controlnet= controlnext, 
    dtype = torch.float32,
    output_type = output_type,
    scheduler= scheduler
).to('cuda')
promt = ' generate an door with high resolution '

image = Image.open('/home/tiennv/trang/chaos/controlnext/khongquantrong/check_images/idx_2/latents_masked.png')
condition_image =  Image.open('/home/tiennv/trang/chaos/controlnext/khongquantrong/check_images/idx_3180/fill_pixel_values.png').convert("RGB")
condition_image= condition_image.resize((256,256))

mask_image = Image.open('/home/tiennv/trang/chaos/controlnext/khongquantrong/check_images/idx_2/mask_pixel_values.png').convert("RGB")
mask_image= mask_image.resize((512,512))


image = pipeline(
    prompt= promt, 
    image =image,fill_image =condition_image,
    mask_image=mask_image, return_dict = False,
    padding_mask_crop= None,
)[0]
((image[0].save(f'/home/tiennv/trang/chaos/controlnext/khongquantrong/examples/out.jpg')))