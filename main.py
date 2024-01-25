import PIL.Image
import pytesseract
import os
import cv2
import numpy as np
import re
from tabulate import tabulate


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


def extract_information(text):
    customer_name_match = re.search(
        r'Customer Name\s*(\w+)', text, re.IGNORECASE)
    account_number_match = re.search(
        r'(?:Contract|Contrack|newee) \w*\D*(\d+)', text, re.IGNORECASE)
    mobile_number_match = re.search(r'Mobile Number\s*(\d+)', text)
    amount_due_match = re.search(
        r'Amount Due date\s*([\d.]+)\s*\(\s*ETB\s*\)', text)

    # Extracted values or None if not found
    return {
        'customer_name': customer_name_match.group(1).strip() if customer_name_match else None,
        'account_number': account_number_match.group(1).strip() if account_number_match else None,
        'mobile_number': mobile_number_match.group(1).strip() if mobile_number_match else None,
        'amount_due': amount_due_match.group(1).strip() if amount_due_match else None
    }


for file in file_names:
    image = cv2.imread(f"./assets/{file}")

    gray = get_grayscale(image)
    thresh = thresholding(gray)
    o = opening(gray)
    c = canny(gray)

    text = pytesseract.image_to_string(c, config=my_config)
    content.append(text)

content_info = []

for text in content:
    extracted_info = extract_information(text)
    content_info.append(extracted_info)

# Define the column headers
headers = ["Customer Name", "Account Number", "Mobile Number", "Amount Due"]
rows = []

rows = []
for info in content_info:
    rows.append([info['customer_name'], info['account_number'],
                info['mobile_number'], info['amount_due']])

# Print the table
print(tabulate(rows, headers=headers, tablefmt="grid"))
