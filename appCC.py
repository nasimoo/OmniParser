import os
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import subprocess
import numpy as np
import base64
import pandas as pd
from utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model
)
from ClickyCC import BBClickBLE

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load models
model_path = 'weights/icon_detect_v1_5/model_v1_5.pt'
som_model = get_yolo_model(model_path)
som_model.to(device)

caption_model_processor = get_caption_model_processor(
    model_name="florence2", 
    model_name_or_path="weights/icon_caption_florence", 
    device=device
)

# Capture card configuration
ffmpeg_cmd = [
    'ffmpeg',
    '-f', 'dshow',
    '-rtbufsize', '100M',
    '-i', 'video=USB3.0 Video',  # Replace with your device name
    '-pix_fmt', 'bgr24',
    '-f', 'rawvideo',
    '-'
]
capture_width = 1920
capture_height = 1080
box_overlay_ratio = max(capture_width, capture_height) / 3200

# Configuration for drawing bounding boxes
draw_bbox_config = {
    'text_scale': 0.8 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}
BOX_THRESHOLD = 0.05


def process_frame(frame, som_model, caption_model_processor, ble_clicker):
    """Process a single frame for OCR, SOM labeling, and BLE clicking."""
    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Perform OCR
    ocr_bbox_rslt, _ = check_ocr_box(
        image=image,
        display_img=False,
        output_bb_format='xyxy',
        easyocr_args={'paragraph': False, 'text_threshold': 0.8},
        use_paddleocr=True
    )
    text, ocr_bbox = ocr_bbox_rslt

    # SOM labeling
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image=image,
        som_model=som_model,
        BOX_TRESHOLD=BOX_THRESHOLD,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=0.7,
        scale_img=False,
        batch_size=16
    )

    # Save parsed content to CSV
    output_csv_path = os.path.join(output_dir, 'parsed_content.csv')
    df = pd.DataFrame(parsed_content_list)
    df['ID'] = range(len(df))
    df['bbox'] = df['coordinates'].apply(lambda coords: str([coords[0], coords[1], coords[2], coords[3]]))
    df = df[['ID', 'content', 'bbox']]
    df.to_csv(output_csv_path, index=False)

    # Click on bounding boxes via BLE
    for row in parsed_content_list:
        bbox = [row['coordinates'][0], row['coordinates'][1], row['coordinates'][2], row['coordinates'][3]]
        ble_clicker.click_on_bbox(bbox)

    print(f"Processed frame and clicked bounding boxes.")


def main():
    # Initialize BLE clicker
    ble_clicker = BBClickBLE(csv_file_path=None, bluetooth_com_port="COM14", baud_rate=115200)

    # Start FFMPEG process
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        while True:
            # Read a single frame
            raw_frame = process.stdout.read(capture_width * capture_height * 3)
            if not raw_frame:
                break
            
            frame = np.frombuffer(raw_frame, np.uint8).reshape((capture_height, capture_width, 3))

            # Process the frame
            process_frame(frame, som_model, caption_model_processor, ble_clicker)

            # Display the frame for debugging
            cv2.imshow("Capture Card Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    finally:
        process.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
