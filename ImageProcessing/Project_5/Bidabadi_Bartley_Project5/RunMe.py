# accepts a single noisy image and denoises using all denoising methods
from time import time
import cv2
import numpy as np
import median_filter_SingleImageInput
import wiener_filter_SingleImageInput
import deepinv as dinv
import torch

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained DRUNet from deepinv (downloads weights on first use)
drunet = dinv.models.DRUNet(pretrained="download", device=device).eval()
drunet_greyscale = dinv.models.DRUNet(pretrained="download", device=device, in_channels=1, out_channels=1).eval()

def load_image_keep_channels(path):
    # preserve number of channels and alpha if present
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # convert 4-channel BGRA -> BGR (drop alpha) for metric comparisons
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def to_uint8(img):
    # convert floats to uint8 if needed (assume values in 0..1 for floats)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img

def compute_metrics(ref_path, test_path):
    A = load_image_keep_channels(ref_path)
    B = load_image_keep_channels(test_path)

    # If either is grayscale image read as 2D, keep as 2D. If one is 2D and the other 3D,
    # convert 3D to grayscale to compare apples-to-apples (or replicate channels)
    if A.ndim == 2 and B.ndim == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    elif A.ndim == 3 and B.ndim == 2:
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

    A = to_uint8(A)
    B = to_uint8(B)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    # data_range for uint8 images
    data_range = 255 if A.dtype == np.uint8 else (A.max() - A.min())

    # PSNR works for both grayscale and color
    psnr_val = psnr(A, B, data_range=data_range)

    # For SSIM: specify channel_axis for multichannel arrays
    if A.ndim == 3:
        ssim_val = ssim(A, B, data_range=data_range, channel_axis=-1)
    else:
        ssim_val = ssim(A, B, data_range=data_range)

    return psnr_val, ssim_val

def DRUNet_denoise(noisy_image, sigma=0.1, device=device):
    # determine if grayscale or color
    if noisy_image.ndim == 2:
        noisy_image = noisy_image[:, :, np.newaxis]
        drunet_used = drunet_greyscale
    elif noisy_image.ndim == 3 and noisy_image.shape[2] == 1:
        noisy_image = noisy_image.repeat(3, axis=2)
        drunet_used = drunet_greyscale
    else:
        drunet_used = drunet

    noisy_rgb = noisy_image
    if noisy_rgb.dtype == np.uint8:
        noisy_rgb = noisy_rgb.astype(np.float32) / 255.0

    x = torch.from_numpy(noisy_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        x_denoised = drunet_used(x, sigma=sigma)

    denoised = x_denoised.squeeze(0).permute(1, 2, 0).cpu().numpy()
    denoised = np.clip(denoised, 0.0, 1.0)

    denoised_uint8 = (denoised * 255.0).round().astype(np.uint8)
    denoised_bgr = denoised_uint8  # if your input was BGR, you can swap channels if needed

    # optionally save, but DON'T return the path
    cv2.imwrite('drunet_denoised.png', denoised_bgr)

    return denoised_bgr

def denoise_image(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not read the image.")
        return
    # Apply each method and time each

    start_time = time()
    # Apply Median Filter
    median_denoised = median_filter_SingleImageInput.median_filter_single(image_path)
    median_time = time() - start_time
    print(f"finished median denoisning. time: {median_time:.2f} seconds")
    
    start_time = time()
    # Apply Wiener Filter
    wiener_denoised = wiener_filter_SingleImageInput.apply_wiener_filter_single(image_path)
    wiener_time = time() - start_time
    print(f"finished wiener denoisning. time: {wiener_time:.2f} seconds")
    start_time = time()
    # Apply DRUNet Denoising
    drunet_denoised = DRUNet_denoise(img, sigma=0.1, device=device)
    drunet_time = time() - start_time
    print(f"finished DRUNet denoisning. time: {drunet_time:.2f} seconds")

    # # calculate metrics
    # median_psnr, median_ssim = compute_metrics(image_path, median_denoised)
    # wiener_psnr, wiener_ssim = compute_metrics(image_path, wiener_denoised)
    # drunet_psnr, drunet_ssim = compute_metrics(image_path, 'drunet_denoised.png')
    # print(f"Median Filter - PSNR: {median_psnr:.2f}, SSIM: {median_ssim:.4f}")
    # print(f"Wiener Filter - PSNR: {wiener_psnr:.2f}, SSIM: {wiener_ssim:.4f}")
    # print(f"DRUNet - PSNR: {drunet_psnr:.2f}, SSIM: {drunet_ssim:.4f}")

    # Display results
    cv2.imshow("Original Noisy Image", img)
    cv2.imshow(f"Median Denoised TIME: {median_time:.2f} seconds", cv2.imread(median_denoised))
    cv2.imshow(f"Wiener Filter Denoised TIME: {wiener_time:.2f} seconds", cv2.imread(wiener_denoised))
    cv2.imshow(f"DRUNet Denoised TIME: {drunet_time:.2f} seconds", drunet_denoised)

    return median_denoised, wiener_denoised, drunet_denoised

if __name__ == "__main__":
    image_path = input("Enter the noisy image path: ").strip()
    denoise_image(image_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
