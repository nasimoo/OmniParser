import cv2
import os
import time
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64
import pandas as pd
from utils import (
    check_ocr_box,
    get_som_labeled_img,
    get_caption_model_processor,
    get_yolo_model
)

@dataclass
class ProcessingConfig:
    """Configuration for single frame capture and processing."""
    capture_card_index: int = 1
    resolution: Tuple[int, int] = (1920, 1080)
    output_dir: str = "output"
    model_path: str = "weights/icon_detect_v1_5/model_v1_5.pt"
    box_threshold: float = 0.05
    batch_size: int = 16
    caption_model_name: str = "florence2"
    caption_model_path: str = "weights/icon_caption_florence"


class SingleFrameProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup processing components
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_logging()
        
        # Initialize models
        self.som_model = self._load_som_model()
        self.caption_model_processor = self._load_caption_model()

    def setup_logging(self):
        log_path = self.output_dir / 'processing.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_som_model(self):
        """Load and configure the SOM model."""
        try:
            model = get_yolo_model(self.config.model_path)
            model.to(self.device)
            self.logger.info(f"SOM model loaded successfully to {self.device}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load SOM model: {e}")
            raise

    def _load_caption_model(self):
        """Load and configure the caption model processor."""
        try:
            processor = get_caption_model_processor(
                model_name=self.config.caption_model_name,
                model_name_or_path=self.config.caption_model_path,
                device=self.device
            )
            self.logger.info("Caption model processor loaded successfully")
            return processor
        except Exception as e:
            self.logger.error(f"Failed to load caption model processor: {e}")
            raise

    def capture_and_process_frame(self) -> Dict[str, Any]:
        """Capture and process a single frame."""
        results = {}
        start_time = time.time()
        
        try:
            # Initialize capture
            cap = cv2.VideoCapture(self.config.capture_card_index)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open capture card at index {self.config.capture_card_index}")
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            
            # Capture one frame and wait for stabilization
            self.logger.info("Capturing warmup frame and waiting for stabilization (2 seconds)...")
            ret, _ = cap.read()
            time.sleep(2)  # Wait for stabilization
            
            # Capture the actual frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError("Failed to capture frame")
            
            # Save frame
            frame_path = self.output_dir / "raw_frame.png"
            cv2.imwrite(str(frame_path), frame)
            results['frame_path'] = str(frame_path)
            
            # Process frame
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                str(frame_path),
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.8},
                use_paddleocr=True
            )
            text, ocr_bbox = ocr_bbox_rslt
            
            # Calculate box overlay config
            box_overlay_ratio = max(frame.shape[:2]) / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }
            
            # Run detection
            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                str(frame_path),
                self.som_model,
                BOX_TRESHOLD=self.config.box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=0.7,
                scale_img=False,
                batch_size=self.config.batch_size
            )
            
            # Save results
            output_image_path = self.output_dir / "labeled_frame.png"
            decoded_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
            decoded_image.save(output_image_path)
            
            output_csv_path = self.output_dir / "bbox_content.csv"
            df = pd.DataFrame(parsed_content_list)
            df['ID'] = range(len(df))  # Add ID column
            df.to_csv(output_csv_path, index=False)
            
            # Store results
            results.update({
                'ocr_text': text,
                'ocr_bbox': ocr_bbox,
                'labeled_image_path': str(output_image_path),
                'csv_path': str(output_csv_path),
                'label_coordinates': label_coordinates,
                'processing_time': time.time() - start_time
            })
            
            self.logger.info(f"Frame processed in {results['processing_time']:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            results['error'] = str(e)
            return results
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            torch.cuda.empty_cache()

def main():
    config = ProcessingConfig()
    processor = SingleFrameProcessor(config)
    results = processor.capture_and_process_frame()
    print("Processing results:", results)

if __name__ == "__main__":
    main()
