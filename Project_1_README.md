# Image Processing Project 1

## Overview
This project processes a set of images to classify them as either "day" or "night" based on histogram similarity between color and grayscale versions. It uses OpenCV and Python for image analysis.

## How to Use
1. Place your images in a folder (e.g., `Images/ImageSet1`).
2. Run the script `Bartley_Nathan_Project1.py`.
3. When prompted, enter the path to your image folder (e.g., `Images/ImageSet1`).
4. The script will classify each image and display it with its classification.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install requirements with:
```
pip install opencv-python numpy matplotlib
```

## File Structure
- `Bartley_Nathan_Project1.py` - Main script for image classification
- `Images/` - Folder containing image sets
- `Project_1_README.md` - This documentation file

## Example
```
Enter Folder Path: Images/ImageSet1
Images/ImageSet1/O_EK000017.JPG is classified as day
Images/ImageSet1/O_EK000721.JPG is classified as night
...
```

## Author
Nathan Bartley
