# use frequency domain filtering to find patterns in a fabric and make the lighting uniform across the image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def uniform_lighting(image, ksize = 101):
    # big kernel for light smoothing
    img = image.astype(np.float32)
    illum = cv2.GaussianBlur(img, (ksize, ksize), 0)
    out = img - illum + illum.mean()  # remove shading, keep average level
    out -= out.min()
    out /= (out.max() + 1e-6)
    return (out * 255).astype(np.uint8)

def main():
    # Load the image
    # image_path = r"C:\ImageProcessing\Project_3\Assignment\Original.tif"
    image_path = input("Enter Image Path: ")
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("Error: Image not found.")
        return

    image = uniform_lighting(original)
    # plt.imshow(image, cmap='gray')
    # plt.title('Uniform Lighting Image')
    # plt.show()
    F = fft2(image)
    F_shifted = fftshift(F)
    magnitude_spectrum = np.log(np.abs(F_shifted) + 1)
    # plt.figure(figsize=(12, 6))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('Magnitude Spectrum')
    # plt.show()

    # 7 highest peaks in the magnitude spectrum
    flat = magnitude_spectrum.flatten()
    indices = np.argpartition(flat, -7)[-7:]
    peaks = np.array(np.unravel_index(indices, magnitude_spectrum.shape)).T
    print("Peak coordinates (y, x):", peaks)

    mask = np.zeros_like(magnitude_spectrum, dtype=bool)
    mask[peaks[:, 0], peaks[:, 1]] = 1.0
    # plt.imshow(mask, cmap='gray')
    # plt.title('Mask for Filtering')
    # plt.show()
    
    # Apply the mask and inverse FFT
    F_shifted_filtered = F_shifted * mask
    F_filtered = ifftshift(F_shifted_filtered)
    image_filtered = np.abs(ifft2(F_filtered))
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)
    # plt.imshow(image_filtered, cmap='gray')
    # plt.title('Filtered Image')
    # plt.show()

    # Show mask, uniform lighting, and filtered images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(mask, cmap='gray')
    axs[0].set_title('Mask for Filtering')
    axs[0].axis('off')
    axs[1].imshow(image, cmap='gray')
    axs[1].set_title('Uniform Lighting Image')
    axs[1].axis('off')
    axs[2].imshow(image_filtered, cmap='gray')
    axs[2].set_title('Filtered Image')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()