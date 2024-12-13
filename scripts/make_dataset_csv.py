# import json
# import csv
# import os
# import pandas as pd

# # Directories for images and JSON files
# image_folder = '/home/tiennv/trang/chaos/controlnext/data/images'  # Replace with your image folder path
# json_folder = '/home/tiennv/trang/chaos/controlnext/data/labels'  # Replace with your JSON folder path
# fill_image_folder = '/mnt/Datadrive/datasets/deepfurniture/Identities'  # Folder containing fill images
# output_csv = '/home/tiennv/trang/chaos/controlnext/data/datatest/dataset_deepfinetune_v2_data.csv'  # Path to save the CSV file

# # Ensure the output directory exists
# output_dir = os.path.dirname(output_csv)
# os.makedirs(output_dir, exist_ok=True)

# # Step 1: Create CSV from JSON files
# with open(output_csv, 'w', newline='') as csvfile:
#     # Add columns for image paths, bounding boxes, masks, and fill image paths
#     fieldnames = ['image_path', 'bbox', 'mask', 'fill_image_path']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()

#     # Loop through each JSON file in the JSON folder
#     for json_file in os.listdir(json_folder):
#         if json_file.endswith('.json'):
#             json_path = os.path.join(json_folder, json_file)

#             # Load JSON data
#             with open(json_path, 'r') as file:
#                 data = json.load(file)
            
#             # Extract image ID from JSON and locate the corresponding image
#             image_id = data.get("scene", {}).get("sceneTaskID", "")
#             image_path = os.path.join(image_folder, f"{image_id}.jpg")  # Assuming images are in .jpg format
            
#             # Extract `identityID` from the `instances` list
#             instances = data.get("instances", [])
#             for instance in instances:
#                 identity_id = instance.get("identityID", None)  # Check if `identityID` exists in the instance
#                 if identity_id is not None:
#                     fill_image_path = os.path.join(fill_image_folder, f"{identity_id}.jpg")
#                 else:
#                     fill_image_path = "N/A"  # Use "N/A" if `identityID` is missing
                
#                 # Extract bounding boxes and masks
#                 if 'boundingBox' in instance:
#                     bbox = instance['boundingBox']
#                     bbox_str = f"{bbox['xMin']},{bbox['yMin']},{bbox['xMax']},{bbox['yMax']}"
                    
#                     # Check if segmentation data is available
#                     mask = instance.get('segmentation', [])
#                     if mask:  # Skip if mask is None or empty
#                         mask_str = ','.join(map(str, mask))  # Convert segmentation list to a string
                        
#                         # Write each bounding box and mask as a separate row
#                         writer.writerow({
#                             'image_path': image_path,
#                             'bbox': bbox_str,
#                             'mask': mask_str,
#                             'fill_image_path': fill_image_path
#                         })

# print("CSV file created successfully.")

# # Step 2: Remove invalid rows from the CSV file
# try:
#     # Load the dataset
#     dataset = pd.read_csv(output_csv)

#     # Check for rows where the 'mask' column contains NaN or None
#     invalid_rows = dataset[dataset['mask'].apply(lambda x: pd.isna(x) or x == "None")].index

#     # Drop rows with invalid values in 'mask' column
#     dataset.drop(index=invalid_rows, inplace=True)

#     # Save the updated dataset back to the original file
#     dataset.to_csv(output_csv, index=False)
#     print(f"Rows with invalid values in 'mask' column have been removed.")
#     print(f"Updated dataset saved to the file: {output_csv}")

# except Exception as e:
#     print(f"An error occurred during cleaning: {e}")



import os
import cv2
import pandas as pd

# Chuyển đổi từ định dạng YOLO (x, y, w, h) sang (xmin, ymin, xmax, ymax)
def xywh2xyxy(x, y, w, h, img_width, img_height):
    try:
        xmin = (x - w / 2) * img_width
        ymin = (y - h / 2) * img_height
        xmax = (x + w / 2) * img_width
        ymax = (y + h / 2) * img_height
        return int(xmin), int(ymin), int(xmax), int(ymax)
    except Exception as e:
        print(f"Error in xywh2xyxy: {e}")
        return None, None, None, None

# Hàm tạo CSV từ thư mục chứa ảnh và label
def create_csv_from_labels(images_folder_path, labels_folder_path, class2choose, save_csv_path):
    try:
        data = []
        # Duyệt qua tất cả các file trong thư mục labels
        for label_file in os.listdir(labels_folder_path):
            # Kiểm tra nếu file là label (.txt)
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_folder_path, label_file)
                image_path = os.path.join(images_folder_path, label_file.replace('.txt', '.jpg'))

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                # Đọc kích thước ảnh
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image: {image_path}")
                    continue

                img_height, img_width = img.shape[:2]

                # Đọc file label và chuyển đổi bbox
                try:
                    with open(label_path, 'r', encoding='ISO-8859-1') as file:
                        for line in file:
                            class_id, x, y, w, h = map(float, line.strip().split())
                            xmin, ymin, xmax, ymax = xywh2xyxy(x, y, w, h, img_width, img_height)
                            if int(class_id) in class2choose:  # Kiểm tra lớp có trong danh sách chọn
                                data.append({"image_path": image_path, "bbox": f"{xmin},{ymin},{xmax},{ymax}"})
                except Exception as e:
                    print(f"Error reading label file {label_path}: {e}")
                    continue

        # Lưu kết quả vào file CSV
        df = pd.DataFrame(data)
        df.to_csv(save_csv_path, index=False)
        print(f"CSV file saved to {save_csv_path}")
    except Exception as e:
        print(f"Error in create_csv_from_labels: {e}")

# Đường dẫn tới thư mục ảnh và thư mục label (cùng thư mục)
images_folder_path = '/home/data/data'
labels_folder_path = '/home/data/data'

# Các lớp bạn muốn giữ lại trong CSV (ví dụ: 0-17)
class2_choose = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# Đường dẫn lưu file CSV
save_csv_path = '/home/data/data_high_quality.csv'

# Bắt đầu quá trình
print("Step 1: Creating CSV from labels...")
create_csv_from_labels(images_folder_path, labels_folder_path, class2_choose, save_csv_path)

print("Process completed!")