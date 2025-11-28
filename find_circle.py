import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread(r"C:/Users/arnau/Documents/DTU/34753_intro_robotics/circle2.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocess
gray = cv2.medianBlur(gray, 5)

# Detect circles
circles = cv2.HoughCircles(gray,
                           cv2.HOUGH_GRADIENT,
                           dp=1.2,
                           minDist=40,
                           param1=100,
                           param2=20,
                           minRadius=5,
                           maxRadius=200)

# Process detection
if circles is not None:
    circles = np.uint16(np.around(circles))
    # Since you only have 1 circle, take the first one
    x, y, r = circles[0][0]

    print(f"Center of circle: x={x}, y={y}")
    print(f"Radius: {r}")

    # Draw circle and center
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

# Show result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
