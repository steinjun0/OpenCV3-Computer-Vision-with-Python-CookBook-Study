from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
cap = cv2.VideoCapture(0)   # 0: default camera
#cap = cv2.VideoCapture("test.mp4") #동영상 파일에서 읽기
 
while cap.isOpened():
    # 카메라 프레임 읽기
    success, frame = cap.read()
    if success:
        # 프레임 출력
        cv2.imshow('Camera Window', frame)
 
        # ESC를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if (key == 27): 
            break
 
cap.release()
cv2.destroyAllWindows()
"""

def showImage():
    imgfile = 'OpenCV\\test1.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.title('model')
    plt.show()

showImage()

img = cv2.imread('OpenCV\\test1.jpg',1)

cv2.namedWindow("Test Image", cv2.WINDOW_NORMAL)
cv2.imshow('Test Image', img)
cv2.waitKey(0)

cv2.destroyAllWindows()


cv2.imwrite('OpenCV\\test2.jpg',img)