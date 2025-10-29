### Nathan Bartley
### Project 2
### ECE 5367 Image Processing

### Card Orientation and cropping 

import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolve2d(image, kernel):

    kernel = kernel.astype(np.float32)
    k_h, k_w = kernel.shape
    h,w = image.shape
    #convolution of single channel
    convolved_channel = np.zeros((h - k_h + 1, w - k_w + 1), dtype=np.float32)
    for i in range(h - k_h + 1):
        for j in range(w - k_w + 1):
            region = image[i:i + k_h, j:j + k_w]
            convolved_channel[i, j] = np.sum(region * kernel)
    return convolved_channel

def rotate_image(image, angle):
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def calculate_angle(mag, angle_map_deg):
    mask = mag.astype(bool)
    theta = (angle_map_deg+90) %180 - 90  # map angles to -90..90
    if np.any(mask):
        bins = np.linspace(-90, 90, 181)
        hist, _ = np.histogram(theta[mask], bins=bins, weights=mag[mask])
        centers = (bins[:-1] + bins[1:]) * 0.5
        peak = centers[np.argmax(hist)]
        plt.bar(centers, hist, width=1.0); plt.title('Angle Histogram'); plt.xlabel('Angle (degrees)'); plt.ylabel('Weighted Count'); plt.show()
        # refine: take angles near the peak and use a robust statistic
        win = 5.0
        sel = mask & (np.abs(theta - peak) <= win)
        rot_angle_deg = float(np.median(theta[sel])) if np.any(sel) else float(peak)
    else:
        rot_angle_deg = 0.0
    return rot_angle_deg

def crop_image(oriented_edges, original_image):
    # Find image-background background-image transitions
    ys, xs = np.where(oriented_edges > 0)
    # leftmost edge pixel
    i_min = np.argmin(xs)
    leftmost = (ys[i_min], xs[i_min])

    i_max = np.argmax(xs)
    rightmost = (ys[i_max], xs[i_max])

    # topmost edge pixel
    j_min = np.argmin(ys)
    topmost = (ys[j_min], xs[j_min])

    j_max = np.argmax(ys)
    bottommost = (ys[j_max], xs[j_max])

    cropped = original_image[topmost[0]+3:bottommost[0]+3, leftmost[1]+3:rightmost[1]+3]
    return cropped

def main():
    image_path = input("Enter Image Path: ")   
    print("Processing image:", image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)                       # Loads as grayscale
    plt.imshow(img, cmap='gray'); plt.title('Original Image'); plt.axis('off'); plt.show()
    # Constant Kernels
    blurnel = np.ones((5,5))/25                                       # 5x5 averaging kernel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])          # Sobel kernel for x direction
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])          # Sobel kernel for y direction
    # Convolutions. Blur, Sobel X, Sobel Y,
    blurred_img = convolve2d(img, blurnel)
    plt.imshow(blurred_img, cmap='gray'); plt.title('Blurred Image'); plt.axis('off'); plt.show()
    Gx = convolve2d(blurred_img, sobel_x)
    Gy = convolve2d(blurred_img, sobel_y)

    mag = np.hypot(Gx, Gy)
    edges = mag / np.max(mag)  # Normalize to 0..1  

    threshold = 0.3
    edges = (edges > threshold).astype(np.uint8)  # Binary edge map
    angle_map = np.arctan2(Gy, Gx)  # Angle in radians
    angle_map_deg = np.degrees(angle_map)   # Convert to degrees

    angle = calculate_angle(mag, angle_map_deg)
    print(f"Calculated rotation angle: {angle:.2f} degrees")
    rotated_edges = rotate_image(edges, angle)
    rotated_final = rotate_image(img, angle)
    cropped_final = crop_image(rotated_edges, rotated_final)

    # Plot original and final images side by side #
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(cropped_final, cmap='gray')
    axs[1].set_title('Cropped Image')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()
    plt.imshow(angle_map_deg, cmap='gray'); plt.title('Angle Map (degrees)'); plt.axis('off'); plt.show()
    plt.imshow(rotated_final, cmap='gray'); plt.title('Rotated Image'); plt.axis('off'); plt.show()
    plt.imshow(cropped_final, cmap='gray'); plt.title('Cropped Image'); plt.axis('off'); plt.show()
    plt.imshow(Gx.astype(np.float32), cmap='gray'); plt.title('Gx'); plt.axis('off'); plt.show()
    plt.imshow(Gy.astype(np.float32), cmap='gray'); plt.title('Gy'); plt.axis('off'); plt.show()
    plt.imshow(edges, cmap='gray'); plt.title('Gradient Magnitude (0..1)'); plt.axis('off'); plt.show()
    plt.imshow(rotated_edges, cmap='gray'); plt.title('Rotated Image'); plt.axis('off'); plt.show()

if __name__ == "__main__":
    main()