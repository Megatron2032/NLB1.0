import cv2


img = cv2.imread("2.jpg")
img=cv2.resize(img,(320,375))
cv2.imwrite("imput.bmp", img)
