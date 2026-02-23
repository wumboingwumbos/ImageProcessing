import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

def TimeOfDay(image_path, plt_show=False):
    threshold = 10000                                                   # Threshold for histogram standard deviation to classify as day or night
    hist = img_histograms(image_path)
    if plt_show:
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
    if np.std(hist) > threshold:
        label = "Night     saturation std: " + str(np.std(hist))
    else:
        label = "Day     saturation std: " + str(np.std(hist))
    return label

def classify_images(folder_path):                                       # Classifies all .JPG images in the specified folder path
    image_paths = glob.glob(os.path.join(folder_path, '*.JPG'))
    for image_path in image_paths:
        classification = TimeOfDay(image_path)
        cv2.imshow(classification, cv2.imread(image_path))
        cv2.waitKey(0)

def crop_watermark(image_path):                                          # Crops out the bottom 50 pixels of the image to remove watermark
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)                       # Loads as BGR
    height, width, _ = img.shape
    cropped_img = img[0:height-50, 0:width]                              # Crop 50 pixels from the bottom
    return cropped_img

def img_histograms(image_path, channel=1):                               # channel: 0-H, 1-S, 2-V
    img = crop_watermark(image_path)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                       # Convert to HSV
    match channel:                                                       # Create histogram for specified channel
        case 0:
            hist = cv2.calcHist([img_HSV], [0], None, [256], [0,256])
        case 1:
            hist = cv2.calcHist([img_HSV], [1], None, [256], [0,256])    #Found to be best indicator
        case 2:
            hist = cv2.calcHist([img_HSV], [2], None, [256], [0,256])
        case _:
            raise ValueError("Channel must be 0 (H), 1 (S), or 2 (V)")
    return hist.flatten()

if __name__ == "__main__":
    print("Day/Night Classification Is Image Window Title")
    print("Close Image Window To Proceed To Next Image")
    folder_path = input("Enter Folder Path: ")
    classify_images(folder_path)