import PIL.Image
import pytesseract
import os
import cv2
import numpy as np


def get_files_in_folder(path):
    if not os.path.exists(path):
        return []
    files = os.listdir(path)
    return files


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


path = "./assets"
file_names = get_files_in_folder(path)

my_config = r"--psm 6 --oem 3"
content = []

for file in file_names:
    image = cv2.imread(f"./assets/{file}")

    gray = get_grayscale(image)
    thresh = thresholding(gray)
    o = opening(gray)
    c = canny(gray)

    text = pytesseract.image_to_string(c, config=my_config)
    content.append(text)

for read in content:
    print(read)
