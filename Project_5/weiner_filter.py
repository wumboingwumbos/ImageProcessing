import os
import cv2
import numpy as np
from scipy.signal import wiener

def apply_wiener_filter_to_folder():
    folder_path = input("Enter the folder path containing images: ").strip()

    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a valid directory.")
        return

    output_folder = os.path.join(folder_path, "wiener_filtered_output")
    os.makedirs(output_folder, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Size of the local window used by Wiener filter (e.g., 5x5)
    wiener_window = (5, 5)

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_ext:
            continue

        img_path = os.path.join(folder_path, filename)
        print(f"Processing: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {filename}, skipping.")
            continue

        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0

        # Apply Wiener filter channel by channel (scipy.signal.wiener is 2D)
        filtered_channels = []
        for c in range(img_float.shape[2]):
            channel = img_float[:, :, c]
            # Wiener filter expects 2D array, window controls local statistics
            filtered_channel = wiener(channel, mysize=wiener_window)
            # Clip to valid range just in case
            filtered_channel = np.clip(filtered_channel, 0.0, 1.0)
            filtered_channels.append(filtered_channel)

        filtered_img = np.stack(filtered_channels, axis=2)

        # Convert back to uint8
        filtered_img_uint8 = (filtered_img * 255.0).astype(np.uint8)

        save_path = os.path.join(output_folder, f"wiener_{filename}")
        cv2.imwrite(save_path, filtered_img_uint8)
        print(f"Saved: {save_path}")

    print("\nDONE! Wiener-filtered images saved in:", output_folder)


if __name__ == "__main__":
    apply_wiener_filter_to_folder()
