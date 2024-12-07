
from safetensors.torch import load_file, save_file
import torch
import os
import json


def get_dtype_training(dtype):
    if dtype == 'bf16':
        return torch.bfloat16
    if dtype == 'fp16':
        return torch.float16
    else:
        return torch.float32

def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
    if not load_weight_increasement:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)




def log_data_make_loss_high(loss_value, idx_list, loss_threshold, save_log_path):
    import os
    import json
    """
    Logs a list of indices where the loss value exceeds the specified threshold.

    Args:
        loss_value (float): The current loss value.
        idx_list (list): A list of indices to check and log.
        loss_threshold (float): The threshold above which indices will be logged.
        save_log_path (str): Path to the JSON file for saving logged indices.

    Description:
        If the loss value exceeds the threshold, the function ensures that all indices in `idx_list` 
        are added to the log file located at `save_log_path`. The log file is a JSON list, and duplicate
        indices will not be added.
    """
    if loss_value > loss_threshold:
        # Load existing log if the file exists, or initialize an empty list
        if os.path.exists(save_log_path):
            with open(save_log_path, 'r') as f:
                idx_check_list = json.load(f)
        else:
            idx_check_list = []

        # Append indices from idx_list to the log if they are not already present
        for idx in idx_list:
            if idx not in idx_check_list:
                idx_check_list.append(idx)

        # Save the updated log to the file
        with open(save_log_path, 'w') as f:
            json.dump(idx_check_list, f)


                



def check_image_in_dataset(dataset, data_csv_path, json_data_path, number_idx, output_image_folder_path):
    
    import os
    import random
    import json
    from torchvision import transforms
    from PIL import Image
    import pandas as pd
    
    """
    Check and save samples from the dataset, with images restored and saved in separate folders.
    
    Args:
        dataset: Initialized dataset object.
        data_csv_path: Path to the CSV file containing image information, bbox, and mask.
        json_data_path: Path to the JSON file containing a list of indices to plot.
        number_idx: Number of indices to randomly select and save.
        output_image_folder_path: Folder to save the output images.
    """
    # Load the JSON file containing the list of indices
    with open(json_data_path, 'r') as f:
        available_idxs = json.load(f)
    
    # Randomly select `number_idx` indices from the JSON file
    # selected_idxs = random.sample(available_idxs, min(number_idx, len(available_idxs)))
    # selected_idxs = [3180]
    selected_idxs= [2,3,4,5,6]

    # Read the CSV file
    data = pd.read_csv(data_csv_path)
    
    for idx in selected_idxs:
        # Retrieve information from the CSV file
        row = data.iloc[int(idx)]
        image_path = row['image_path']
        bbox = row['bbox']
        # mask = row['mask']
        mask = None
        fill_image_path = row.get('fill_image_path', None)

        # Generate data using the `make_data` method
        make_data = dataset.make_data(image_path, fill_image_path, bbox, mask)

        # Create a subfolder for the current index
        idx_folder = os.path.join(output_image_folder_path, f"idx_{idx}")
        os.makedirs(idx_folder, exist_ok=True)

        # Save the output images
        for key, tensor_image in make_data.items():
            if key in ['latents_target', 'latents_masked', 'fill_pixel_values', 'mask_pixel_values']:
                # Restore the tensor image to the original range
                tensor_image = (tensor_image * 0.5) + 0.5  # Undo normalization
                tensor_image = tensor_image.clamp(0, 1)  # Ensure values are in range [0, 1]

                # Convert tensor to PIL image
                pil_image = transforms.ToPILImage()(tensor_image)
                
                # Define the output file name
                output_path = os.path.join(idx_folder, f"{key}.png")
                
                # Save the image
                pil_image.save(output_path)
                print(f"Saved {key} for idx {idx} to {output_path}")

    print(f"Successfully saved {len(selected_idxs)} samples to {output_image_folder_path}.")


