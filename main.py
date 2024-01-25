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


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


path = "./assets"
file_names = get_files_in_folder(path)

my_config = r"--psm 6 --oem 3"
content = []

for file in file_names:
    image = cv2.imread(f"./assets/{file}")
    edited_image = opening(image)
    text = pytesseract.image_to_string(edited_image, config=my_config)
    content.append(text)

for read in content:
    print(read)
