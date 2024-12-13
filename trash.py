from lightning.pytorch import Trainer
from trainer import LitSDCATDF_v1  # Thay bằng lớp LightningModule thực tế của bạn

# Đường dẫn đến checkpoint
checkpoint_path = "/home/tiennv/trang/chaos/trash/output/checkpoints/model-epoch=1193-train_loss=0.0003.ckpt"
model = LitSDCATDF_v1.load_from_checkpoint(checkpoint_path, map_location="cpu")
# Truy cập unet
unet = model.unet
import torch
# # In thông tin cấu hình hoặc các tham số của unet
# print("UNet Config:", unet.config)  # Nếu unet có thuộc tính config
# print("UNet State Dict Keys:", unet.state_dict().keys())

# Kiểm tra siêu tham số đã được nạp
# Lưu trọng số của unet
unet_weights_path = "/home/tiennv/trang/chaos/trash/output/diffusion_pytorch_model.bin"
torch.save(unet.state_dict(), unet_weights_path)
print(f"UNet weights saved to {unet_weights_path}")