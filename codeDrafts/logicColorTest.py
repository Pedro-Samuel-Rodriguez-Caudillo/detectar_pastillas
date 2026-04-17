import cv2
import numpy as np 
from PIL import Image
import scipy.misc

input_img = cv2.imread("pills/orange_circle.jpg")
imgtest = 255 - input_img
img = cv2.resize(input_img, (640, 480))
cv2.imshow('image', img)
input_img_cpy = img.copy()
cv2.waitKey(0)

colors = []
# Convert RGB/ BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

lower_green = np.array([40,20,10])
upper_green = np.array([90,255,255])


lower_blue = np.array ([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv,lower_blue ,upper_blue )

 #create a mask for red color
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

lower_turqoise = np.array([70, 100, 120])
upper_turqoise = np.array([120, 150, 255])
mask_turqoise = cv2.inRange(hsv,lower_turqoise ,upper_turqoise )

#orange
lower_orange = np.array([15, 180, 170])
upper_orange = np.array([20, 255, 220])
mask_orange = cv2.inRange(hsv,lower_orange,upper_orange)
mask_orange2 = cv2.inRange(hsv,lower_orange,upper_orange)

lower_grey = np.array([0, 0, 80])
upper_grey = np.array([255, 25, 160])
mask_grey = cv2.inRange(hsv,lower_grey ,upper_grey )

lower_black = np.array([0,0,0])
upper_black = np.array([359,255,25])
mask_black = cv2.inRange(hsv, lower_black, upper_black)







 
# Display filtered image
cv2.imshow('mask_red', mask_green)
cv2.waitKey(0)

# find contours in the red mask
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_turqoise, _ = cv2.findContours(mask_turqoise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_grey, _ = cv2.findContours(mask_turqoise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_black, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



shapes, hierarchygreen = cv2.findContours(image = mask_green, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
shapes, hierarchyred = cv2.findContours(image=mask_red, mode=cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
shapes, hierarchyblue = cv2.findContours(image = mask_blue, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
rect_areas = []





 
# Draw detected contour in input image






for cnt in contours_red:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        isred = True
        colors.append('red')
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


for cnt in contours_green:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 2000:
        isgreen = True
        colors.append('green')
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'green!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for cnt in contours_blue:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 2000:
        isblue = True
        colors.append('blue')
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'blue!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for cnt in contours_orange:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 2000:
        isblue = True
        colors.append('orange')
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'orange!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for cnt in contours_turqoise:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 2000:
        isblue = True
        colors.append('turqoise')
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'turqoise!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


for cnt in contours_black:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 2000:
        isblue = True
        colors.append('black')
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'black!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for cnt in contours_grey:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 2000:
        isblue = True
        colors.append('grey')
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'gray!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
 
 
 

 

# Dispay contour
cv2.imshow('image',img) 
print(colors)



cv2.waitKey(0)


