import cv2
import numpy as np 
input_img = cv2.imread("pills/green_square.jpg")
input_img2 = cv2.imread("pills/grey_circle.jpg")
img = cv2.resize(input_img, (640, 480))
img2 = cv2.resize(input_img2, (640, 480))
#cv2.imshow('image', img)
input_img_cpy = img.copy()
input_img_cpy2 = img2.copy()



# Convert RGB/ BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)

#green
lower_green = np.array([40,20, 10])
upper_green = np.array([90, 255, 255])
mask_green = cv2.inRange(hsv,lower_green,upper_green)

#turqoise
lower_turqoise = np.array([70, 100, 120])
upper_turqoise = np.array([120, 150, 255])
mask_turqoise = cv2.inRange(hsv,lower_turqoise ,upper_turqoise )

#orange
lower_orange = np.array([15, 180, 170])
upper_orange = np.array([20, 255, 220])
mask_orange = cv2.inRange(hsv,lower_orange,upper_orange)
mask_orange2 = cv2.inRange(hsv2,lower_orange,upper_orange)

#blue
lower_blue = np.array ([100, 80, 60])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv,lower_blue ,upper_blue )

#grey
lower_grey = np.array([0, 0, 80])
upper_grey = np.array([255, 25, 160])
mask_grey = cv2.inRange(hsv,lower_grey ,upper_grey )
mask_grey2 = cv2.inRange(hsv2,lower_grey ,upper_grey )

#black
lower_black = np.array([0, 0, 0])
upper_black = np.array([360, 255, 25])
mask_black = cv2.inRange(hsv,lower_black ,upper_black )


 
# Display filtered image
cv2.imshow('mask_green', mask_green)
#cv2.imshow('mask_green2', mask_orange2)



# find contours in the red mask
#contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw detected contour in input image
#contour_red_cap = cv2.drawContours(input_img_cpy, contours_red, -1, (255, 0, 255), 3)

# find contours in the green mask
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw detected contour in green image
contour_green_cap = cv2.drawContours(input_img_cpy, contours_green, -1, (255, 0, 255), 3)



# find contours in the blue mask
#contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw detected contour in input image
#contour_blue_cap = cv2.drawContours(input_img_cpy, contours_blue, -1, (255, 0, 255), 3)

# find contours in the turqoise mask
#contours_turqoise, _ = cv2.findContours(mask_turqoise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw detected contour in input image
#contour_turqoise_cap = cv2.drawContours(input_img_cpy, contours_turqoise, -1, (255, 0, 255), 3)
 
# find contours in the orange mask
#contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw detected contour in input image
#contour_orange_cap = cv2.drawContours(input_img_cpy, contours_orange, -1, (255, 0, 255), 3)

#grey
#contours_grey, _ = cv2.findContours(mask_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#contour_grey_cap = cv2.drawContours(input_img_cpy, contours_grey, -1, (255, 0, 255), 3)

#white
#contours_white, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#contour_white_cap = cv2.drawContours(input_img_cpy, contours_white, -1, (255, 0, 255), 3)



#contours_grey2, _ = cv2.findContours(mask_grey2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#contour_grey_cap2 = cv2.drawContours(input_img_cpy2, contours_grey2, -1, (255, 0, 255), 3)

i = 0
for contour in contours_green:
  
    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue
  
    if cv2.contourArea(contour) < 3000:
        continue


    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.05 * cv2.arcLength(contour, True), True)
      
    # using drawContours() function
    cv2.drawContours(input_img, [contour], 0, (0, 0, 255), 5)
  
    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
  
    # putting shape name at center of each shape
    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
    elif len(approx) == 4:
        cv2.putText(img, 'Quadrilateral', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
    elif len(approx) == 5:
        cv2.putText(img, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
    elif len(approx) == 6:
        cv2.putText(img, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
    else:
        cv2.putText(img, 'circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
# Dispay contour
cv2.imshow('contour_blue_cap', img)
cv2.imshow('contour', contour_green_cap)
#cv2.imshow('contour_blue_cap2', contour_grey_cap2)


cv2.waitKey(0)
