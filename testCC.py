import cv2

def open_capture_card_feed(capture_card_index=0):
    # Open the capture card at the specified index
    cap = cv2.VideoCapture(capture_card_index, cv2.CAP_DSHOW)  # Use DirectShow backend for Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Desired width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Desired height
    cap.set(cv2.CAP_PROP_FPS, 30)  # Desired FPS

    if not cap.isOpened():
        print(f"Unable to open capture card at index {capture_card_index}.")
        return

    # Print capture card properties for debugging
    print(f"Capture Card Properties:")
    print(f" - Frame Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f" - Frame Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f" - FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    print("Press 'ESC' to close the feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Display the feed
        cv2.imshow("Capture Card Feed", frame)

        # Exit the feed when ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for ESC
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Call the function
open_capture_card_feed(capture_card_index=0)  # Adjust the index if needed
