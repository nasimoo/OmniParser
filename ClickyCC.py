import cv2
import serial
import time
import csv
import ast

# BLE Configuration
BLUETOOTH_COM_PORT = "COM4"
BAUD_RATE = 115200

# Screen and Display Configuration
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# CSV File Path
CSV_FILE_PATH = "output/gmailoptions_bbox_content.csv"


# Basic offset
BASE_X_OFFSET = -10
Y_OFFSET = -10

def calculate_x_offset(x):
    """Calculate progressive X offset based on X coordinate"""
    try:
        offset = BASE_X_OFFSET
        offset += -2 * (x // 400)  # Adjust offset calculation as needed
        return int(offset)  # Ensure the result is an integer
    except Exception as e:
        print(f"Error in calculate_x_offset with x={x}: {e}")
        return 0  # Default to 0 if there's an issue


def calculate_y_offset(y):
    """Calculate progressive Y offset based on Y coordinate"""
    try:
        offset = Y_OFFSET
        offset += 1 * (y // 300)  # Adjust offset calculation as needed
        return int(offset)  # Ensure the result is an integer
    except Exception as e:
        print(f"Error in calculate_y_offset with y={y}: {e}")
        return 0  # Default to 0 if there's an issue


    
class ClickableDS:
    def __init__(self, com_port, baud_rate):
        self.ser = self.initialize_serial_connection(com_port, baud_rate)

    def initialize_serial_connection(self, com_port, baud_rate):
        try:
            ser = serial.Serial(com_port, baud_rate, timeout=1)
            time.sleep(2)
            print("Connected to BLE device")
            return ser
        except Exception as e:
            print(f"Failed to connect to BLE device: {e}")
            return None

    def send_command(self, cmd):
        if self.ser:
            try:
                self.ser.write((cmd + "\n").encode('utf-8'))
                print(f"Sent: {cmd}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to send command: {e}")

    def move_to_origin(self):
        self.send_command("ABS:0,0")
        time.sleep(0.5)

    def send_keyboard_input(self, text):
        """
        Sends a string of text as keyboard input.
        """
        if self.ser:
            try:
                self.send_command(f"kbd:{text}")
                print(f"Keyboard input sent: {text}")
            except Exception as e:
                print(f"Failed to send keyboard input: {e}")

    def press_f11(self):
        """
        Sends the F11 key press command.
        """
        self.send_command("f11")
        print("F11 key pressed.")

def find_bounding_box_by_id(specific_id):
    try:
        with open(CSV_FILE_PATH, "r") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                if row["ID"] == specific_id:
                    bbox = ast.literal_eval(row["bbox"])
                    print(f"Found bbox {bbox} for ID: {specific_id}")

                    x_min = int(bbox[0] * SCREEN_WIDTH)
                    y_min = int(bbox[1] * SCREEN_HEIGHT)
                    x_max = int(bbox[2] * SCREEN_WIDTH)
                    y_max = int(bbox[3] * SCREEN_HEIGHT)

                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                    return x_center, y_center

            print(f"Bounding box with ID '{specific_id}' not found in the CSV.")
            return None

    except FileNotFoundError:
        print(f"CSV file not found: {CSV_FILE_PATH}")
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
    return None
def main():
    # Initialize BLE connection
    click_ds = ClickableDS(BLUETOOTH_COM_PORT, BAUD_RATE)
    time.sleep(1)

    # Move to origin before typing
    click_ds.move_to_origin()
    time.sleep(1)

    try:
        # Type the text
        text = "1235 how are you my name is nasim nice to meet you 4455665656"
        click_ds.send_keyboard_input(text)
        print("Text typed successfully.")
    except Exception as e:
        print(f"Failed to type the text: {e}")

    if click_ds.ser:
        click_ds.ser.close()


if __name__ == "__main__":
    main()
