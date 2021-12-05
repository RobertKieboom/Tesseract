import re
import os
import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
counter = 0


def recognise_sign(path):
    # Read the image
    img = cv2.imread(path)
    height, width, channels = img.shape

    # Scale large images down
    maxres = 700
    if height > maxres or width > maxres:
        scale = min(maxres / width, maxres / height)
        w = int(width * scale)
        h = int(height * scale)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LANCZOS4)

    # Find circles in the image
    cimg = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1.0, 50)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # For all circles found
        for i in circles[0,:]:
            x1, y1, x2, y2 = i[0] - i[2], i[1] - i[2], i[0] + i[2], i[1] + i[2]

            # Draw debug info
            cv2.circle(img, (i[0], i[1]), i[2], (0,0,255), 2)

            # Crop to the bounding box of the circle
            cropped_image = img[y1:y2, x1:x2]

            # Create black text on white background
            lower = np.array([0,0,0])           # Lower range (black)
            upper = np.array([100,100,100])     # Upper range (darkish gray)
            mask = cv2.inRange(cropped_image, lower, upper)
            mask = (255-mask)

            # Convert image to RGB format, because tesseract expects this
            rgb_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # Optical Character Recognition
            config = "--oem 3 --psm 6 outputbase digits"
            text = pytesseract.image_to_string(rgb_mask, config=config);

            # Extract the number from the string
            numbers = re.findall(r'\d+', text)
            if len(numbers) > 0:
                return numbers[0]

            # If no number found, try the next cicrle
    else:
        # No circles found, draw message
        cv2.putText(img, "No Circles", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 8)
        cv2.putText(img, "No Circles", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Nothing found, show debug info
    global counter
    cv2.imshow("Show_" + str(counter), img)
    counter = counter + 1
    return None


# For all images in this directory
for file in os.listdir("."):
    if file.endswith(".jpg") or file.endswith(".png"):
        text = recognise_sign(file)
        print(f"{file}: {text}")

cv2.waitKey(0)
