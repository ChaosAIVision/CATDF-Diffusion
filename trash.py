# from diffusers import  UNet2DConditionModel

# unet = UNet2DConditionModel.from_pretrained(
#                 'botp/stable-diffusion-v1-5-inpainting', subfolder ='unet'
#             )

# unet.to('cuda')

# from huggingface_hub import HfApi
# api = HfApi()

# # Upload all the content from the local folder to your remote Space.
# # By default, files are uploaded at the root of the repo
# from huggingface_hub import HfApi

# api = HfApi()

# # Tải lên toàn bộ thư mục
# api.upload_folder(
#     folder_path="/home/checkpoint",  # Đây phải là một thư mục
#     repo_id="Nguyentruong031203/model_v1",
#     path_in_repo="checkpoints_sample_72/",  # Tên thư mục trong repo
#     repo_type="dataset"  # Hoặc "model"
# )
# from lightning.pytorch import Trainer
# from trainer import LitSDCATDF_v1  # Thay bằng lớp LightningModule thực tế của bạn

# # Đường dẫn đến checkpoint
# checkpoint_path = "/home/checkpoint/model-epoch=24-train_loss=0.0030.ckpt"

# # Tải lại mô hình từ checkpoint
# model = LitSDCATDF_v1.load_from_checkpoint(checkpoint_path, map_location="cpu")
# # Truy cập unet
# unet = model.unet
# import torch
# # # In thông tin cấu hình hoặc các tham số của unet
# # print("UNet Config:", unet.config)  # Nếu unet có thuộc tính config
# # print("UNet State Dict Keys:", unet.state_dict().keys())

# # Kiểm tra siêu tham số đã được nạp
# # Lưu trọng số của unet
# unet_weights_path = "/home/checkpoint/unet.bin"
# torch.save(unet.state_dict(), unet_weights_path)
# print(f"UNet weights saved to {unet_weights_path}")

# Tải lại mô hình từ checkpoint

from lightning.pytorch import Trainer
from trainer import LitSDCATDF_v1  # Thay bằng lớp LightningModule thực tế của bạn

# Đường dẫn đến checkpoint
checkpoint_path = "/home/save_checkpoint/final_checkpoint.ckpt"
model = LitSDCATDF_v1.load_from_checkpoint(checkpoint_path, map_location="cpu")
# Truy cập unet
unet = model.unet
import torch
# # In thông tin cấu hình hoặc các tham số của unet
# print("UNet Config:", unet.config)  # Nếu unet có thuộc tính config
# print("UNet State Dict Keys:", unet.state_dict().keys())

# Kiểm tra siêu tham số đã được nạp
# Lưu trọng số của unet
unet_weights_path = "/home/weight_pretrain/unet_train/diffusion_pytorch_model.bin"
torch.save(unet.state_dict(), unet_weights_path)
print(f"UNet weights saved to {unet_weights_path}")