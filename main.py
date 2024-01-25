import PIL.Image
import pytesseract


text = pytesseract.image_to_string(PIL.Image.open("./assets/photo.jpg"))

print(text)
