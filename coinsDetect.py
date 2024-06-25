import cv2
import numpy as np
import cvzone
from cvzone.ColorModule import ColorFinder

totalMoney = 0
myColorFinder = ColorFinder(True)
hsvVals = {'hmin': 120, 'smin': 243, 'vmin': 178, 'hmax': 0, 'smax': 255, 'vmax': 255}

def preprocessing(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply morphological operations to enhance edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

# IP address of your phone's camera
phone_ip = '192.168.28.160'

# Video capture from phone's camera
cap = cv2.VideoCapture(f'http://{phone_ip}:8080/video')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame from camera.")
        break

    # Resize the frame
    frame = cv2.resize(frame, (500, 300))
    
    # Preprocess the frame
    imgPre = preprocessing(frame)
    
    # Find contours
    imageContours, conFound = cvzone.findContours(frame, imgPre, minArea=20)

    totalMoney = 0
    if conFound:
        for contour in conFound:
            area = contour['area']
            perimeter = cv2.arcLength(contour['cnt'], True)
            circularity = 4 * np.pi * (area / (perimeter ** 2))

            # Approximate contour
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * perimeter, True)

            if len(approx) > 5:
                print(area)
                x, y, w, h = contour['bbox']
                imgCrop = frame[y:y + h, x:x + w]
                imgColor, mask = myColorFinder.update(imgCrop, hsvVals)
                whitePixelCount = cv2.countNonZero(mask)

                if circularity > 0.7:  
                    if area < 2200:
                        totalMoney += 1
                    elif 2250 < area  < 2500:
                        totalMoney += 5
                    elif 2550 < area < 3000:
                        totalMoney += 2
                    else:
                        totalMoney += 10

    print("Total Money:", totalMoney)

    image = cvzone.stackImages([frame, imgPre, imageContours], 2, 1)
    cvzone.putTextRect(image, f'Rs.{totalMoney}', (50, 50), scale=2, thickness=2, offset=10)
    
    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()