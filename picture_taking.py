import cv2
import time

cap = cv2.VideoCapture(1)
img_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save
    if key == ord('s'):
        filename = f"img_{img_id:03d}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved", filename)
        img_id += 1

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
