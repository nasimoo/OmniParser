import csv
import pyautogui
import ast
import time
from PIL import ImageGrab
import cv2
import numpy as np
from clicky import ScreenAutomation

# Step 1: Click on Google Chrome (ID 18 in parsed_content_1.csv)
automation_chrome = ScreenAutomation("output/parsed_content_1.csv")
if automation_chrome.find_and_click("15", "Google."):
    print("Google Chrome opened successfully.")
    time.sleep(2)  # Wait for Chrome to open

    # Step 2: Open a new tab with Ctrl + T
    pyautogui.hotkey("ctrl", "t")
    print("Opened a new tab in Google Chrome.")
    time.sleep(1)

    # Step 3: Click on "G Search Google or type a URL,1" in parsed_content.csv
    automation_search = ScreenAutomation("output/parsed_content.csv")
    if automation_search.find_and_click("1", "G Search Google or type a URL"):
        print("Search bar clicked. Typing URL...")
        pyautogui.typewrite("youtube.com")
        pyautogui.press("enter")
        print("Navigated to YouTube.")
