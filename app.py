# Refactored script from notebook
# Ensure all dependencies are installed before running
import io
import os
import torch
from ultralytics import YOLO
from PIL import Image
import time
import base64
import pandas as pd
from utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model
)

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Failed to create output directory: {e}")
        raise

# Set device to CUDA if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load models
model_path = 'weights/icon_detect_v1_5/model_v1_5.pt'
som_model = get_yolo_model(model_path)
som_model.to(device)
print(f"Model loaded to {device}")

# Load caption model processor
caption_model_processor = get_caption_model_processor(
    model_name="florence2", 
    model_name_or_path="weights/icon_caption_florence", 
    device=device
)

# Image path
image_path = 'imgs/image.png'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path)
image_rgb = image.convert('RGB')
print(f"Image size: {image.size}")

# Box overlay configuration
box_overlay_ratio = max(image.size) / 3200
draw_bbox_config = {
    'text_scale': 0.8 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}
BOX_THRESHOLD = 0.05

# Perform OCR and SOM labeling
try:
    start_time = time.time()
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_path, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold': 0.8}, 
        use_paddleocr=True
    )
    text, ocr_bbox = ocr_bbox_rslt

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_path, 
        som_model, 
        BOX_TRESHOLD=BOX_THRESHOLD, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox, 
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text, 
        use_local_semantics=True, 
        iou_threshold=0.7, 
        scale_img=False, 
        batch_size=16  # Reduced batch size for memory optimization
    )
    print(f"SOM labeling completed in {time.time() - start_time:.2f} seconds.")
    
except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory. Try reducing the batch size or using CPU.")
    som_model.to('cpu')
    torch.cuda.empty_cache()
    raise

# Decode and save labeled image
output_image_path = os.path.join(output_dir, 'labeled_image.png')
try:
    decoded_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    decoded_image.save(output_image_path)
    print(f"Labeled image saved to {output_image_path}")
except Exception as e:
    print(f"Failed to save labeled image: {e}")
    raise

# Save parsed content as a CSV
output_csv_path = os.path.join(output_dir, 'parsed_content.csv')
try:
    df = pd.DataFrame(parsed_content_list)
    df['ID'] = range(len(df))
    df.to_csv(output_csv_path, index=False)
    print(f"Parsed content saved to {output_csv_path}")
except Exception as e:
    print(f"Failed to save parsed content: {e}")
    raise
