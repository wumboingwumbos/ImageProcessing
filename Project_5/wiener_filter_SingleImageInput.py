import os
import cv2
import numpy as np
from scipy.signal import wiener

def apply_wiener_filter_single():
    img_path = input("Enter the image path: ").strip()

    if not os.path.isfile(img_path):
        print("Error: The provided path is not a valid file.")
        return

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    ext = os.path.splitext(img_path)[1].lower()

    if ext not in valid_ext:
        print("Error: Unsupported file type.")
        return

    print(f"Processing: {img_path}")

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not read the image.")
        return

    # Convert to float
    img_float = img.astype(np.float32) / 255.0

    # Wiener filter window size
    wiener_window = (5, 5)

    # Process each color channel separately
    filtered_channels = []
    for c in range(img_float.shape[2]):
        channel = img_float[:, :, c]
        filtered_channel = wiener(channel, mysize=wiener_window)
        filtered_channel = np.clip(filtered_channel, 0.0, 1.0)
        filtered_channels.append(filtered_channel)

    filtered_img = np.stack(filtered_channels, axis=2)

    # Convert back to 0–255 uint8
    filtered_img_uint8 = (filtered_img * 255.0).astype(np.uint8)

    # Save next to original file
    folder = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    save_path = os.path.join(folder, f"wiener_{filename}")

    cv2.imwrite(save_path, filtered_img_uint8)
    print(f"Saved Wiener-filtered image as: {save_path}")


if __name__ == "__main__":
    apply_wiener_filter_single()
