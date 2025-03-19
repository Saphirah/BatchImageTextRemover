import json
import os
import urllib.request
import cv2
import numpy as np
import pytesseract
from pyinpaint import Inpaint
from simple_lama_inpainting import SimpleLama

simple_lama = SimpleLama()

def download_image(image):
    print('Downloading image from:', image["url"])
    #Check if exists
    if not os.path.exists('source/' + str(image["id"]) + '.jpg'):
        urllib.request.urlretrieve(image["url"], 'source/' + str(image["id"]) + '.jpg')

def process_image(imageData):
    # Read the image
    image_path = "source/" + str(imageData["id"]) + ".jpg"
    img = cv2.imread(image_path)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Invert the image
    rgb_img = cv2.bitwise_not(rgb_img)
    #Treshold white
    _, rgb_img = cv2.threshold(rgb_img, 10, 255, cv2.THRESH_BINARY)
    border_size = 200
    rgb_img = cv2.copyMakeBorder(
        rgb_img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # Get bounding box data using pytesseract
    data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)

    # Create new Black White image
    height, width = img.shape[:2]
    mask = np.zeros((height, width, 1), dtype=np.uint8)

    # Loop through each detected word (or text block)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i] - border_size, data['top'][i] - border_size, data['width'][i], data['height'][i])
        if x <= 0 or y <= 0 or w <= 0 or h <= 0:
            continue
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Fill the bounding box with white color
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Save the resulting image
    inpainted_img = simple_lama(img, mask)
    output_path = "target/" + str(imageData["id"]) + ".jpg"
    cv2.imwrite(output_path, inpainted_img)
    print(f"Saved annotated image as {output_path}")

if __name__ == '__main__':
    json_data = json.loads(open('images.json').read())
    print('Downloading', len(json_data), 'images')
    for image in json_data:
        download_image(image)
        process_image(image)
        break
