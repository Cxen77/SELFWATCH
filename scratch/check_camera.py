import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera 0 could not be opened.")
else:
    ret, frame = cap.read()
    if not ret:
        print("Camera 0 opened but could not read frame.")
    else:
        print("Camera 0 is working. Frame shape:", frame.shape)
cap.release()
