import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

statedict = torch.load('/home/tiennv/trang/chaos/controlnext/weight_training/checkpoints/epoch-epoch=189.ckpt')
# print(statedict['state_dict'].keys())
model_state_dict = statedict['state_dict']

prefix_controlnext = 'controlnext.'
prefix_unet = 'unet.'

unet_weights = {}
controlnext_weights = {}

for key, value in model_state_dict.items():
    if key.startswith(prefix_controlnext):
        new_key = key[len(prefix_controlnext):]
        controlnext_weights[new_key] = value
    elif key.startswith(prefix_unet):
        new_key = key[len(prefix_unet):]
        unet_weights[new_key] = value

torch.save(unet_weights, '/home/tiennv/trang/chaos/controlnext/khongquantrong/save_weight/unet.bin')
torch.save(controlnext_weights, '/home/tiennv/trang/chaos/controlnext/khongquantrong/save_weight/controlnext.bin')

print("Unet and ControlNext weights have been split and saved separately with prefixes removed.")
