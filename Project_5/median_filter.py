import cv2
import os

def median_filter_folder():
    # Ask user for folder path
    folder_path = input("Enter the folder path containing images: ").strip()

    # Validate path
    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a valid directory.")
        return

    # Output folder
    output_folder = os.path.join(folder_path, "filtered_output")
    os.makedirs(output_folder, exist_ok=True)

    # Allowed image extensions
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Process each file
    for filename in os.listdir(folder_path):
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in valid_ext:
            img_path = os.path.join(folder_path, filename)
            print(f"Processing: {img_path}")

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Skipped unreadable file {filename}")
                continue

            # Apply median filtering (kernel size = 3)
            filtered = cv2.medianBlur(img, 3)

            # Save output
            save_path = os.path.join(output_folder, f"median_{filename}")
            cv2.imwrite(save_path, filtered)
            print(f"Saved filtered image: {save_path}")

    print("\nDONE! Filtered images saved in:", output_folder)


if __name__ == "__main__":
    median_filter_folder()
