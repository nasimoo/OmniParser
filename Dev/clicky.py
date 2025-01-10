import csv
import pyautogui
import ast
import time
from PIL import ImageGrab
import cv2
import numpy as np

class BBClick:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.screen_width, self.screen_height = pyautogui.size()

    def is_bbox_visible(self, bbox):
        x_min = int(bbox[0] * self.screen_width)
        y_min = int(bbox[1] * self.screen_height)
        x_max = int(bbox[2] * self.screen_width)
        y_max = int(bbox[3] * self.screen_height)

        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        roi = screen_np[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return False

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        non_zero_count = cv2.countNonZero(gray_roi)

        return non_zero_count > 0

    def click_on_bbox(self, bbox, click_delay=0.5):
        x_min = bbox[0] * self.screen_width
        y_min = bbox[1] * self.screen_height
        x_max = bbox[2] * self.screen_width
        y_max = bbox[3] * self.screen_height

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        pyautogui.moveTo(x_center, y_center)
        pyautogui.click()
        time.sleep(click_delay)

    def find_and_click(self, specific_id, specific_content=None):
        try:
            with open(self.csv_file_path, "r") as csv_file:
                reader = csv.DictReader(csv_file)

                for row in reader:
                    if row["ID"] == specific_id and (specific_content is None or row["content"] == specific_content):
                        bbox = ast.literal_eval(row["bbox"])
                        print(f"Sanity Check: Found bbox {bbox}, ID: {row['ID']}, Content: {row['content']}")

                        if self.is_bbox_visible(bbox):
                            print("Bounding box is visible. Performing click...")
                            self.click_on_bbox(bbox)
                        else:
                            print("Bounding box is not visible on the current screen. Skipping click.")
                        return True

                print(f"Element with ID '{specific_id}' and content '{specific_content}' not found in the CSV.")
                return False

        except FileNotFoundError:
            print(f"CSV file not found: {self.csv_file_path}")
        except Exception as e:
            print(f"Failed to read CSV file: {e}")
        return False