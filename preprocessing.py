import cv2
import numpy as np
import os

def process_single_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

    enhanced_img = cv2.convertScaleAbs(img_gray - laplacian)

    leaf_canny3 = cv2.Canny(enhanced_img, 450, 550)

    kernel = np.ones((2,2), np.uint8)
    img_dilatation_result = cv2.dilate(leaf_canny3, kernel, iterations=1)
    
    img_sharp = cv2.resize(img_dilatation_result,(128,96), interpolation=cv2.INTER_AREA)

    image_flat = img_sharp.flatten()
    
    return image_flat