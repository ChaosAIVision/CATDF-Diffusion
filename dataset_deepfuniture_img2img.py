
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import get_dtype_training
import random
import numpy as np
import pandas
import os
from pathlib import Path
import pandas as pd
from utils import check_image_in_dataset
from PIL import ImageOps


def collate_fn_embedding(examples):
    
    latents_target = torch.stack([example["latents_target"] for example in examples])
    latents_target = latents_target.to(memory_format=torch.contiguous_format)
    
    latents_masked = torch.stack([example['latents_masked'] for example in examples])
    latents_masked = latents_masked.to(memory_format=torch.contiguous_format)
 
    fill_pixel_values = torch.stack([example['fill_pixel_values'] for example in examples])
    fill_pixel_values = fill_pixel_values.to(memory_format=torch.contiguous_format)
    
    mask_pixel_values = torch.stack([example['mask_pixel_values'] for example in examples])
    mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format)
    
    # index_fill_image = torch.stack([example['index_fill_image'] for example in examples])
    # index_fill_image = index_fill_image.to(memory_format=torch.contiguous_format)
    

    return {
        "latents_target": latents_target,
        "latents_masked": latents_masked,
        'fill_pixel_values': fill_pixel_values,
        'mask_pixel_values': mask_pixel_values,
        # 'index_fill_image': index_fill_image
    }


def collate_fn(examples):
    latents_target = torch.stack([example["latents_target"] for example in examples])
    latents_target = latents_target.to(memory_format=torch.contiguous_format)
    
    latents_masked = torch.stack([example['latents_masked'] for example in examples])
    latents_masked = latents_masked.to(memory_format=torch.contiguous_format)
 
    fill_pixel_values = torch.stack([example['fill_pixel_values'] for example in examples])
    fill_pixel_values = fill_pixel_values.to(memory_format=torch.contiguous_format)
    
    mask_pixel_values = torch.stack([example['mask_pixel_values'] for example in examples])
    mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format)
    
    encoder_hidden_state = torch.stack([example['encoder_hidden_state'] for example in examples])
    encoder_hidden_state = encoder_hidden_state.to(memory_format=torch.contiguous_format)
    
    # index_fill_image = torch.stack([example['index_fill_image'] for example in examples])
    # index_fill_image = index_fill_image.to(memory_format=torch.contiguous_format)
    
    return {
        "latents_target": latents_target,
        "latents_masked": latents_masked,
        'fill_pixel_values': fill_pixel_values,
        'mask_pixel_values': mask_pixel_values,
        'encoder_hidden_state': encoder_hidden_state,
        # 'index_fill_image': index_fill_image

    }


class Deepfurniture_Dataset_V1(Dataset):
    """
    Dataset for Deepfurniture project.
    Includes:
    - Pixel values for latent targets (loss calculation)
    - Masked images
    - Fill images for masked areas
    - Text embeddings
    """

    def __init__(self, data_path, input_type,dtype , image_size=512):
        self.input_type = input_type
        if self.input_type == "raw":
            self.data = pandas.read_csv(data_path)
        else:
            self.data = Path(data_path)
            metadata_file =  self.data / "metadata.parquet"
            if not metadata_file.exists():
                raise FileNotFoundError(f"metadata.parquet not in {self.data}")          
            self.metadata = pd.read_parquet(metadata_file)
    
        self.image_size = image_size
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.dtype = get_dtype_training(dtype)

    def __len__(self):
        if self.input_type == "raw":
            return len(self.data)
        else:
            return len(self.metadata)

    def string_to_list(self, str_annotation):
        """
        Converts a string of annotations to a list of integers.
        """
        if isinstance(str_annotation, float):
            print(f"Converting float to string for processing: {str_annotation}")
            str_annotation = str(int(str_annotation))
        
        return list(map(int, str_annotation.split(',')))

    def get_bboxes_annotations(self, bbox_string):
        """
        Parses bbox string to get bounding box coordinates.
        """
        return self.string_to_list(bbox_string)

    def get_mask_image(self,image,mask_string ):
        width,height, = image.size  
        segmentation_annotation = self.string_to_list(mask_string)
        mask = np.zeros((width, height), dtype=np.uint8)  
        
        curr_pos = 0
        curr_val = 0

        for idx in range(len(segmentation_annotation)):
            curr_count = segmentation_annotation[idx]
            end_pos = curr_pos + curr_count  
            while curr_pos < end_pos:
                row = curr_pos // height
                col = curr_pos % height
                if row < width and col < height:  
                    mask[row, col] = curr_val  
                curr_pos += 1
            curr_val = 1 - curr_val        
        mask_image = Image.fromarray(((mask.T) *255).astype(np.uint8))
        return mask_image

    
    def get_fill_images(self, image, bbox_annotations): # crop image base on mask
        """
        Creates an image with the area inside bbox filled with white.
        """
        xmin, ymin, xmax, ymax = bbox_annotations
        cropped_image = image.crop((xmin, ymin, xmax, ymax))  
        return cropped_image
    
    def get_masked_image(self, image, mask): # get masked image base on mask
        """
        Apply a white overlay to the image based on the mask.

        Args:
            image (PIL.Image.Image): The original image (RGB format).
            mask (np.ndarray): Binary mask (2D numpy array).

        Returns:
            PIL.Image.Image: The resulting image with the mask applied.
        """
        image_array = np.array(image)  # Shape: (H, W, C)
        mask = np.array(mask)

        binary_mask = (mask > 0).astype(np.uint8) * 255

        binary_mask = binary_mask[:, :, None]

        # Create a white overlay (same size as the image)
        white_overlay = np.ones_like(image_array) * 255 
        masked_image_array = np.where(binary_mask == 255, white_overlay, image_array)
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))

        return masked_image
    
    
    def create_bbox_mask(self, image, bbox):
        """
        Creates a black-and-white mask where the region inside the bbox is white,
        and the region outside is black.

        Args:
            image (PIL.Image.Image): The original image (RGB format).
            bbox (list): List of 4 elements [x1, y1, x2, y2] representing the bounding box.

        Returns:
            PIL.Image.Image: Black-and-white mask image with bbox region as white and outside as black.
        """
        # Create an empty black mask with the same size as the input image
        mask = Image.new("L", image.size, color=0)  # "L" mode is for grayscale, black background

        # Draw the bbox region as white (255 intensity)
        x1, y1, x2, y2 = bbox
        for x in range(x1, x2):
            for y in range(y1, y2):
                mask.putpixel((x, y), 255)  # Set bbox region to white

        return mask


    
    def get_masked_image_with_bbox(self, image, bbox): # get masked base on bbox
        """
        Apply a white overlay to a specific bounding box in the image.

        Args:
            image (PIL.Image.Image): The original image (RGB format).
            bbox (list): List of 4 elements [x1, y1, x2, y2] representing the bounding box.

        Returns:
            PIL.Image.Image: The resulting image with the white overlay applied on the bounding box.
        """
        image_array = np.array(image)  # Shape: (H, W, C)
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_array.shape[1], x2), min(image_array.shape[0], y2)
        white_overlay = np.ones((y2 - y1, x2 - x1, 3), dtype=np.uint8) * 255
        masked_image_array = image_array.copy()
        masked_image_array[y1:y2, x1:x2] = white_overlay
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))

        return masked_image
        
    def get_conditional_image(self,image, mask):
        """
        Keep the region inside the mask unchanged and set the rest of the image to white.
        If there are still areas outside the mask that are not fully covered by white,
        set the entire image to white.

        Args:
            image (PIL.Image.Image): The original image (RGB format).
            mask (np.ndarray): Binary mask (2D numpy array).

        Returns:
            PIL.Image.Image: The resulting image with the mask applied.
        """
        image_array = np.array(image)  # Shape: (H, W, C)
        mask = np.array(mask)

        binary_mask = (mask > 0).astype(np.uint8) * 255

        white_overlay = np.ones_like(image_array) * 255
        masked_image_array = np.where(binary_mask[:, :, None] == 255, image_array, white_overlay)
        if not np.all(masked_image_array[binary_mask == 0] == 255):
            masked_image_array = np.ones_like(image_array) * 255

        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        #resize 256 conditional image
        masked_image = masked_image.resize((256, 256), Image.Resampling.LANCZOS)

        return masked_image
    
    from PIL import Image

    def make_fill(self,image, identities, bbox):
        """
        Overlay an identity image onto a specific bounding box region of the given image.

        Args:
            image (PIL.Image.Image): The original image (RGB format).
            identities (PIL.Image.Image): The identity image to overlay.
            bbox (list): List of 4 elements [x1, y1, x2, y2] representing the bounding box.

        Returns:
            PIL.Image.Image: The resulting image with the identity overlay applied.
        """
        # Ensure bbox is valid
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bounding box coordinates")

        # Compute bbox width and height
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Resize the identity image to match the bbox size
        resized_identity = identities.resize((bbox_width, bbox_height), Image.Resampling.LANCZOS)

        # Paste the resized identity onto the original image at the bbox location
        image = image.copy()  # Make a copy to avoid modifying the original image
        image.paste(resized_identity, (x1, y1))

        return image

    def pad_image_bottom(self, image, target_height):
        """
        Pad the bottom of the image to match the target height.

        Args:
            image (PIL.Image.Image): Input image.
            target_height (int): Target height of the image.

        Returns:
            PIL.Image.Image: Image padded at the bottom to the target height.
        """
        width, height = image.size

        # Calculate the amount of padding needed
        pad_height = max(0, target_height - height)

        # Apply padding only to the bottom
        padding = (0, 0, 0, pad_height)  # (left, top, right, bottom)
        padded_image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))  # White padding
        return padded_image

        
    def make_data(self, image_path,fill_image_path, bbox_string, mask_string):
        """
        Generates all required data from an image path and annotations.
        """
        pixel_values = Image.open(image_path).convert("RGB") # image target 
        bboxes_annotation = self.get_bboxes_annotations(bbox_string)
        # mask_image = self.get_mask_image(pixel_values, mask_string)
        mask_image = self.create_bbox_mask(pixel_values, bboxes_annotation)
        # identites = Image.open(fill_image_path).convert("RGB").resize((256,256)) # image of object fill
        identites = self.get_fill_images(pixel_values,bboxes_annotation)
        # fill_images = self.make_fill(pixel_values, identites, bboxes_annotation)
        masked_image = self.get_masked_image_with_bbox(pixel_values, bboxes_annotation)


        pixel_values = self.pad_image_to_target_size(pixel_values, (self.image_size, self.image_size // 2))

        text = "make picture high quality and harmonization "
        return {
            "latents_target": self.image_transforms(pixel_values),
            "latents_masked": self.image_transforms(masked_image),
            "fill_pixel_values": self.image_transforms(identites),
            "mask_pixel_values": self.image_transforms(mask_image),
            "encoder_hidden_state": text
        }
    def load_saved_embeddings(self, idx):
        npz_file = np.load(os.path.join( self.data, "embeddings_data.npz"))
        metadata_df = pd.read_parquet(os.path.join( self.data, "metadata.parquet"))

        if idx is not None:
            if idx < 0 or idx >= len(metadata_df):
                raise ValueError(f"Index {idx} out of range. Must be between 0 and {len(metadata_df) - 1}.")            
            row = metadata_df.iloc[idx]
            return {
                "latents_target": npz_file[row["latents_target_key"]],
                "latents_masked": npz_file[row["latents_masked__key"]],
                "mask_pixel_values": npz_file[row["mask_pixel_values__key"]],
                "encoder_hidden_state": npz_file[row["encoder_hidden_state_key"]],
                "fill_pixel_values": npz_file[row["fill_pixel_values_key"]],
                # "index_fill_image": npz_file[row["index_fill_image_key"]]
                
                
            }

    def __getitem__(self, idx):
        if self.input_type == 'raw':
    
            # item = self.data.iloc[idx]
            item = self.data.iloc[5]
            image_path = item['image_path']
            # fill_image = item['fill_image_path']
            fill_image = None
            bbox_string = item['bbox']
            # mask_string = item['mask']
            mask_string = None 
            batch = self.make_data(image_path,fill_image, bbox_string, mask_string)
            
            batch['index_fill_image'] = torch.tensor([idx])
            
            return batch
        
        else:
            data= self.load_saved_embeddings(idx)
            
            return {
                "latents_target": torch.tensor(data['latents_target']).to(dtype= self.dtype).squeeze(0),
                "latents_masked": torch.tensor(data['latents_masked']).to(dtype= self.dtype).squeeze(0),
                "mask_pixel_values": torch.tensor(data['mask_pixel_values']).to(dtype= self.dtype).squeeze(0),
                "encoder_hidden_state": torch.tensor(data['encoder_hidden_state']).to(dtype= self.dtype).squeeze(0),
                "fill_pixel_values": torch.tensor(data['fill_pixel_values']).to(dtype= self.dtype).squeeze(0),
                # 'index_fill_image': torch.tensor(data['index_fill_image']).to(dtype= self.dtype).squeeze(0)
            }






if __name__ == "__main__":
    import pandas as pd
    data_csv_path = '/home/data/data_high_quality.csv'
    json_data_path = '/home/data/data.json'
    dataset  = dataset = Deepfurniture_Dataset_V1('/home/data/data_high_quality.csv','raw','bf16',512)
    check_image_in_dataset(dataset= dataset, data_csv_path=data_csv_path, json_data_path=json_data_path,number_idx= 5, output_image_folder_path='/home/data')

#    