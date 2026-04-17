import cv2
import numpy as np 
from PIL import Image
input_img = cv2.imread("pills/red_circle.jpg")
imgtest = 255 - input_img
img = cv2.resize(input_img, (640, 480))
#cv2.imshow('image', img)
input_img_cpy = img.copy()
#cv2.waitKey(0)


# Convert RGB/ BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of colors in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

lower_green = np.array([55,20,10])
upper_green = np.array([90,255,255])


lower_blue = np.array ([100, 80, 60])
upper_blue = np.array([130, 255, 255])

lower_turqoise = np.array([70, 100, 120])
upper_turqoise = np.array([120, 150, 255])

lower_orange = np.array([15, 180, 170])
upper_orange = np.array([20, 255, 220])

#create a masks 
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv,lower_blue ,upper_blue )
mask_orange = cv2.inRange(hsv,lower_orange,upper_orange)
mask_turqoise = cv2.inRange(hsv,lower_turqoise ,upper_turqoise )



 
# Display filtered image
#cv2.imshow('mask_red', mask_red)
#cv2.waitKey(0)

# find contours in the masks
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_turqoise, _ = cv2.findContours(mask_turqoise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




shapesBlue, hierarchyblue = cv2.findContours(image = mask_blue, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
shapesGreen, hierarchygreen = cv2.findContours(image = mask_green, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
shapesRed, hierarchyred = cv2.findContours(image=mask_red, mode=cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
shapesOrange, hierarchyorange = cv2.findContours(image=mask_orange, mode=cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
shapesTurqoise, hierarchyturqoise = cv2.findContours(image=mask_turqoise, mode=cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)


rect_areas = []

for cnt in contours_red:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        rect_areas.append(w*h)
        avg_area = np.mean(rect_areas)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
for cnt in contours_red:
    (x,y,w,h) = cv2.boundingRect(cnt)
    cnt_area = w * h
    if cnt_area < 0.5 * avg_area:
        img[y: y + h, x: x+ w] = 0
        input_img_cpy[y: y+h, x: x+w] = 0




 
# Draw detected contour in input image
contour_red_cap = cv2.drawContours(input_img_cpy, shapesRed, -1, (255, 0, 255), 3)
contour_blue_cap = cv2.drawContours(input_img_cpy, shapesBlue, -1, (255,0,255), 3)
contour_green_cap = cv2.drawContours(input_img_cpy, shapesGreen, -1, (255,0,255), 3)
contour_orange_cap = cv2.drawContours(input_img_cpy, shapesOrange, -1, (255, 0, 255), 3)
contour_turqoise_cap = cv2.drawContours(input_img_cpy, shapesTurqoise, -1, (255, 0, 255), 3)





for cnt in contours_red:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        isred = True
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for cnt in contours_green:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        isgreen = True
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'green!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

for cnt in contours_blue:
    contour_area = cv2.contourArea(cnt)
    if contour_area > 1000:
        isblue = True
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'blue!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
 

# Dispay contour
cv2.imshow('image',contour_red_cap) 



cv2.waitKey(0)



