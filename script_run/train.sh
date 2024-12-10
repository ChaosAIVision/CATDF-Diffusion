CUDA_VISIBLE_DEVICES=0  accelerate launch \
 --main_process_port 12345  ../train.py \
 --pretrained_model_name_or_path="botp/stable-diffusion-v1-5-inpainting" \
 --output_dir="/home/save_checkpoint" \
 --resolution=512 \
 --adam_weight_decay 1e-4 \
 --gradient_accumulation_steps 1 \
 --checkpoints_total_limit 5 \
 --num_train_epochs 1000 \
 --train_batch_size=1 \
 --mixed_precision 'bf16' \
 --prediction_type 'sample' \
 --unet_model_name_or_path '/home/weight_pretrain/unet' \
 --dataset_path '/home/data/data_high_quality.csv' \
  --path_to_save_data_embedding '/home/data/embedding_raw' \
  --input_type 'raw' \
  --resume_from_checkpoint 'latest' \
  --save_embeddings_to_npz True \

#   --controlnext_model_name_or_path '/home/tiennv/trang/chaos/controlnext/weight_pretrain/controlnext/controlnet.safetensors' \

  # --save_embeddings_to_npz True \
  # --report_to 'wandb' \

#  --load_unet_increaments '/home/tiennv/chaos/weight_folder/controlnext_weight/mask/unet.safetensors' \
