# erode mask to make thinner
import cv2
import numpy as np

def erode_mask(mask, kernel_size=12, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return eroded
def main():
    # Example usage
    mask_path = 'C:\\ImageProcessing\\Project_4\\rank_masks_bw\\12.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    eroded_mask = erode_mask(mask, kernel_size=3, iterations=5)
    cv2.imwrite('eroded_mask.png', eroded_mask)
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Eroded Mask', eroded_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()