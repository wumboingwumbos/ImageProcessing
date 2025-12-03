import cv2
import os

def median_filter_single():
    # Ask user for image path
    img_path = input("Enter the image path: ").strip()

    # Validate path
    if not os.path.isfile(img_path):
        print("Error: The provided path is not a valid file.")
        return

    # Validate image extension
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    file_ext = os.path.splitext(img_path)[1].lower()

    if file_ext not in valid_ext:
        print("Error: Unsupported file type.")
        return

    print(f"Processing: {img_path}")

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image.")
        return

    # Apply median filtering
    filtered = cv2.medianBlur(img, 3)

    # Save output next to original file
    folder = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    save_path = os.path.join(folder, f"median_{filename}")

    cv2.imwrite(save_path, filtered)
    print(f"Saved filtered image as: {save_path}")


if __name__ == "__main__":
    median_filter_single()
