
import cv2
import numpy as np

# Load and resize image
img = cv2.imread(r"C:\Users\Rits\Desktop\project\assets\five.jpg")
img = cv2.resize(img, (400, 600))
img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gau = cv2.GaussianBlur(img_gray, (5, 5), 2)


def nothing(x):
    pass


# Create a window for the trackbar
cv2.namedWindow("setting")
cv2.createTrackbar("Threshold", "setting", 50, 255, nothing)

while True:
    # Get the current position of the threshold trackbar
    threshold = cv2.getTrackbarPos("Threshold", "setting")

    # Apply Canny edge detection
    result = cv2.Canny(gau, threshold, 255)

    # Dilate the result to make the edges more pronounced
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.dilate(result, kernel, iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area
    sortcon = sorted(contours, key=cv2.contourArea)

    # Reset img_copy for each iteration
    img_copy = img.copy()

    for i, cont in enumerate(sortcon):
        x, y, w, h = cv2.boundingRect(cont)

        # Find the center of each bounding rectangle
        X = x + int(w / 2)
        Y = y + int(h / 2)

        # Draw circle and put text on img_copy
        img_copy = cv2.circle(img_copy, (X, Y), 50, (0, 255, 0), 2)
        img_copy = cv2.putText(img=img_copy, text=str(i), org=(
            X, Y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)

    # Display images
    cv2.imshow("Canny", result)
    cv2.imshow("Original", img_copy)

    # Break the loop when 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Destroy all windows
cv2.destroyAllWindows()
