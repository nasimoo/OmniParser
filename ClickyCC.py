import csv
import ast
import time
import serial
from PIL import ImageGrab
import cv2
import numpy as np

class BBClickBLE:
    def __init__(self, csv_file_path, bluetooth_com_port, baud_rate):
        self.csv_file_path = csv_file_path
        self.screen_width, self.screen_height = 1920, 1080  # Adjust as necessary
        self.bluetooth_com_port = bluetooth_com_port
        self.baud_rate = baud_rate
        self.ser = self.initialize_serial_connection()

    def initialize_serial_connection(self):
        """Initialize the BLE serial connection."""
        try:
            ser = serial.Serial(self.bluetooth_com_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Give BLE some time to settle
            print("Connected to BLE device")
            return ser
        except Exception as e:
            print(f"Failed to connect to BLE device: {e}")
            return None

    def send_command(self, cmd):
        """Send a command string over BLE."""
        if self.ser:
            try:
                self.ser.write((cmd + "\n").encode('utf-8'))
                print(f"Sent: {cmd}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to send command: {e}")

    def is_bbox_visible(self, bbox):
        """Check if the bounding box is visible on the screen."""
        x_min = int(bbox[0] * self.screen_width)
        y_min = int(bbox[1] * self.screen_height)
        x_max = int(bbox[2] * self.screen_width)
        y_max = int(bbox[3] * self.screen_height)

        # Capture the screen
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        roi = screen_np[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return False

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        non_zero_count = cv2.countNonZero(gray_roi)

        return non_zero_count > 0

    def calculate_offset(self, target_x, target_y):
        """Calculate dynamic offsets based on target coordinates."""
        x_offset = (target_x // 100) * -400  # Add 40 for every 100 x-coordinates
        y_offset = (target_y // 100) * 20  # Add 20 for every 100 y-coordinates
        return x_offset, y_offset

    def click_on_bbox(self, bbox, click_delay=0.5):
        """Send a BLE click command based on the bounding box coordinates."""
        x_min = bbox[0] * self.screen_width
        y_min = bbox[1] * self.screen_height
        x_max = bbox[2] * self.screen_width
        y_max = bbox[3] * self.screen_height

        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)

        # Calculate offsets
        x_offset, y_offset = self.calculate_offset(x_center, y_center)
        x_center += x_offset
        y_center += y_offset

        # Send BLE command
        self.send_command(f"CLICK:{x_center},{y_center}")
        time.sleep(click_delay)

    def find_and_click(self, specific_id, specific_content=None):
        """Find an element in the CSV and click its bounding box if visible."""
        try:
            with open(self.csv_file_path, "r") as csv_file:
                reader = csv.DictReader(csv_file)

                for row in reader:
                    if row["ID"] == specific_id and (specific_content is None or row["content"] == specific_content):
                        bbox = ast.literal_eval(row["bbox"])
                        print(f"Found bbox {bbox}, ID: {row['ID']}, Content: {row['content']}")

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

# Example Usage
if __name__ == "__main__":
    csv_file = "output/parsed_content.csv"
    ble_com_port = "COM14"
    baud_rate = 115200

    bb_click_ble = BBClickBLE(csv_file, ble_com_port, baud_rate)

    # Example: Find and click a bounding box by ID
    specific_id = "39"
    specific_content = None  # Set content if needed
    bb_click_ble.find_and_click(specific_id, specific_content)
