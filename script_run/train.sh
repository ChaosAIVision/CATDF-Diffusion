CUDA_VISIBLE_DEVICES=1  accelerate launch --config_file '/home/tiennv/.cache/huggingface/accelerate/default_config.yaml' --main_process_port 12345  ../train.py \
 --pretrained_model_name_or_path="botp/stable-diffusion-v1-5-inpainting" \
 --output_dir="/home/tiennv/trang/chaos/controlnext/weight_training" \
 --resolution=512 \
 --learning_rate=5e-5 \
 --adam_weight_decay 1e-2 \
 --gradient_accumulation_steps 8 \
 --checkpoints_total_limit 5 \
 --checkpointing_steps 500 \
 --num_train_epochs 1000 \
 --train_batch_size=1 \
 --controlnext_scale 1.0 \
 --mixed_precision 'bf16' \
 --prediction_type 'sample'
 --unet_model_name_or_path '/home/tiennv/trang/chaos/controlnext/weight_pretrain/unet_inpainting' \
  --dataset_path '/home/tiennv/trang/chaos/controlnext/data/datatest/data_high_quality.csv' \
  --path_to_save_data_embedding '/home/tiennv/trang/chaos/controlnext/data/datatest/data_high_quality' \
  --input_type 'raw' \
  --resume_from_checkpoint 'latest' \
  # --save_embeddings_to_npz True \

#   --controlnext_model_name_or_path '/home/tiennv/trang/chaos/controlnext/weight_pretrain/controlnext/controlnet.safetensors' \

  # --save_embeddings_to_npz True \
  # --report_to 'wandb' \

#  --load_unet_increaments '/home/tiennv/chaos/weight_folder/controlnext_weight/mask/unet.safetensors' \
