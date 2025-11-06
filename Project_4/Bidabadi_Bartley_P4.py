#!/usr/bin/env python3

import os
import glob
import sys
from typing import final
import numpy as np
import cv2
import matplotlib.pyplot as plt
import card_classifier as cc
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ------------------------- helpers -------------------------

def ensure_3ch(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def crop_warp(img_bgr, quad):
    def dist(a, b): return float(np.linalg.norm(a - b))
    wA = dist(quad[0], quad[1]); wB = dist(quad[2], quad[3])
    hA = dist(quad[0], quad[3]); hB = dist(quad[1], quad[2])
    width  = int(round(max(wA, wB)))
    height = int(round(max(hA, hB)))
    if height < width:
        width, height = height, width
        quad = np.roll(quad, -1, axis=0)
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img_bgr, M, (width, height))

def make_upright(warped):
    def corner_darkness(im, patch_ratio=0.18):
        ph = pw = max(20, int(min(im.shape[:2]) * patch_ratio))
        ph = min(ph, im.shape[0] // 2)
        pw = min(pw, im.shape[1] // 2)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = 255 - th
        tl = np.sum(th[0:ph, 0:pw])
        tr = np.sum(th[0:ph, -pw:])
        br = np.sum(th[-ph:, -pw:])
        bl = np.sum(th[-ph:, 0:pw])
        return [tl, tr, br, bl]
    for _ in range(4):
        s = corner_darkness(warped)
        if s[0] >= max(s[1:]):
            break
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return warped

def choose_card_mask(gray_blur):
    _, bw_otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_adp = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    def pick(mask):
        inv = 255 - mask
        c1 = largest_contour(mask); a1 = 0 if c1 is None else cv2.contourArea(c1)
        c2 = largest_contour(inv);  a2 = 0 if c2 is None else cv2.contourArea(c2)
        return inv if a2 > a1 else mask
    m1 = pick(bw_otsu)
    m2 = pick(bw_adp)
    c1 = largest_contour(m1); a1 = 0 if c1 is None else cv2.contourArea(c1)
    c2 = largest_contour(m2); a2 = 0 if c2 is None else cv2.contourArea(c2)
    return m2 if a2 > a1 else m1

def upright_crop_card(img_bgr):
    img = ensure_3ch(img_bgr)
    H, W = img.shape[:2]
    scale = 1280 / max(H, W) if max(H, W) > 1280 else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = choose_card_mask(blur)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    edges = cv2.Canny(mask, 50, 150)
    cnt = largest_contour(edges)
    if cnt is None or cv2.contourArea(cnt) < 500:
        raise RuntimeError("No card-like contour found.")

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        quad = order_points(approx.reshape(4, 2).astype(np.float32))
    else:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        quad = order_points(box.astype(np.float32))

    warped = crop_warp(img, quad)
    upright = make_upright(warped)
    return upright

# ------------------------- camera capture -------------------------

def capture_from_camera():
    print("Opening camera... Press SPACE to capture, ESC to cancel.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        sys.exit(1)

    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Camera (Press SPACE to capture)", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("Cancelled.")
            break
        elif key == 32:  # SPACE
            captured = frame.copy()
            print("Image captured.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

# ------------------------- main flow -------------------------

def main():
    print("Choose input method:")
    print("1) Read images from folder")
    print("2) Capture image using camera")
    choice = input("Enter 1 or 2: ").strip()

    images = []

    if choice == "1":
        folder = input("Enter the path to the folder containing images: ").strip().strip('"')
        # check if individual file
        if not folder or not os.path.isdir(folder):
            print("Invalid folder.")
            sys.exit(1)

        paths = []
        for ext in VALID_EXTS:
            paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
        paths = sorted(set(paths))
        if not paths:
            print("No images found in folder:", folder)
            sys.exit(0)

        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append((os.path.basename(p), img))
    elif choice == "2":
        img = capture_from_camera()
        if img is not None:
            images.append(("captured_image.jpg", img))
        else:
            sys.exit(0)
    else:
        print("Invalid choice.")
        sys.exit(1)

    print(f"Processing {len(images)} image(s)...")

    for name, img in images:
        try:
            upright = upright_crop_card(img)
            final, rank_name, suit_name = cc.main(upright)
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(cv2.cvtColor(ensure_3ch(img), cv2.COLOR_BGR2RGB))
            ax1.axis('off')
            ax1.set_title(f"Input: {name}")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
            ax2.axis('off')
            ax2.set_title(f"{rank_name} of {suit_name}")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[err] {name}\n{e}")

if __name__ == "__main__":
    main()
