import testcloudapi as tci
import cv2
import numpy as np 
import pandas as pd
import streamlit as st
import PIL 
from pathlib import Path
import os




def identify(name):
    shape = "notsquare"
    input_img = cv2.imread(name)
    img = cv2.resize(input_img, (640, 480))
    #cv2.imshow('image', img)
    input_img_cpy = img.copy()
    shape = ''



    # Convert RGB/ BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

    #blue
    lower_blue = np.array ([100, 80, 60])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv,lower_blue ,upper_blue )

    #grey
    lower_grey = np.array([0, 0, 80])
    upper_grey = np.array([255, 25, 160])
    mask_grey = cv2.inRange(hsv,lower_grey ,upper_grey )

    #black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([360, 255, 25])
    mask_black = cv2.inRange(hsv,lower_black ,upper_black )

    #establishing contours
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


    colors = []


    
    # Draw detected contour in input image

    # find contours in the red mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw detected contour in input image
    contour_red_cap = cv2.drawContours(input_img_cpy, contours_red, -1, (255, 0, 255), 3)

    # find contours in the green mask
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw detected contour in green image
    contour_green_cap = cv2.drawContours(input_img_cpy, contours_green, -1, (255, 0, 255), 3)



    # find contours in the blue mask
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw detected contour in input image
    contour_blue_cap = cv2.drawContours(input_img_cpy, contours_blue, -1, (255, 0, 255), 3)

    # find contours in the turqoise mask
    contours_turqoise, _ = cv2.findContours(mask_turqoise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw detected contour in input image
    contour_turqoise_cap = cv2.drawContours(input_img_cpy, contours_turqoise, -1, (255, 0, 255), 3)
    
    # find contours in the orange mask
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw detected contour in input image
    contour_orange_cap = cv2.drawContours(input_img_cpy, contours_orange, -1, (255, 0, 255), 3)

    #grey
    contours_grey, _ = cv2.findContours(mask_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_grey_cap = cv2.drawContours(input_img_cpy, contours_grey, -1, (255, 0, 255), 3)





    for cnt in contours_red:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1000:
            isred = True
            colors.append('RED')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'RED', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    for cnt in contours_green:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 2000:
            isgreen = True
            colors.append('GREEN')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'GREEN!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for cnt in contours_blue:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 2000:
            isblue = True
            colors.append('BLUE')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'BLUE!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for cnt in contours_orange:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 2500:
            isblue = True
            colors.append('ORANGE')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'ORANGE!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for cnt in contours_turqoise:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 2500:
            isblue = True
            colors.append('TURQOISE')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'TURQOISE!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    for cnt in contours_black:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 2500:
            isblue = True
            colors.append('BLACK')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'BLACK!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for cnt in contours_grey:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 2500:
            isblue = True
            colors.append('WHITE')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'WHITE!!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
 

    if colors[0] == 'GREEN':
        i = 0
        for contour in contours_green:
        
            # here we are ignoring first counter because 
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue
        
            if cv2.contourArea(contour) < 2000:
                continue


            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.04 * cv2.arcLength(contour, True), True)
            
            # using drawContours() function
            cv2.drawContours(input_img, [contour], 0, (0, 0, 255), 5)
        
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
        
            # putting shape name at center of each shape
            if len(approx) == 3:
                shape = 'TRIANGLE'
                cv2.putText(img, 'TRIANGLE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                shape = 'SQUARE'
                cv2.putText(img, 'SQUARE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                shape = 'PENTAGON'
                cv2.putText(img, 'PENTAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                shape = 'HEXAGON'
                cv2.putText(img, 'HEXAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                shape = 'ROUND'
                cv2.putText(img, 'ROUND', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                


                
    elif colors[0] == 'RED':
        i = 0
        for contour in contours_red:
        
            # here we are ignoring first counter because 
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue
        
            if cv2.contourArea(contour) < 3000:
                continue


            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.033* cv2.arcLength(contour, True), True)
            
            # using drawContours() function
            cv2.drawContours(input_img, [contour], 0, (0, 0, 255), 5)
        
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
        
            # putting shape name at center of each shape
            if len(approx) == 3:
                shape = 'TRIANGLE'
                cv2.putText(img, 'TRIANGLE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                shape = 'SQUARE'
                cv2.putText(img, 'SQUARE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                shape = 'PENTAGON'
                cv2.putText(img, 'PENTAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                shape = 'HEXAGON'
                cv2.putText(img, 'HEXAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                shape = 'ROUND'
                cv2.putText(img, 'ROUND', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                

    elif colors[0] == 'BLUE':
        i = 0
        for contour in contours_blue:
        
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
                shape = 'TRIANGLE'
                cv2.putText(img, 'TRIANGLE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                shape = 'SQUARE'
                cv2.putText(img, 'SQUARE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                shape = 'PENTAGON'
                cv2.putText(img, 'PENTAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                shape = 'HEXAGON'
                cv2.putText(img, 'HEXAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                shape = 'ROUND'
                cv2.putText(img, 'ROUND', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    elif colors[0] == 'WHITE':
        i = 0
        for contour in contours_grey:
        
            # here we are ignoring first counter because 
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue
        
            if cv2.contourArea(contour) < 3000:
                continue


            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.04 * cv2.arcLength(contour, True), True)
            
            # using drawContours() function
            cv2.drawContours(input_img, [contour], 0, (0, 0, 255), 5)
        
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
        
            # putting shape name at center of each shape
            if len(approx) == 3:
                shape = 'TRIANGLE'
                cv2.putText(img, 'TRIANGLE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                shape = 'SQUARE'
                cv2.putText(img, 'SQUARE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                shape = 'PENTAGON'
                cv2.putText(img, 'PENTAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                shape = 'HEXAGON'
                cv2.putText(img, 'HEXAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                shape = 'ROUND'
                cv2.putText(img, 'ROUND', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif colors[0] == 'ORANGE':
        i = 0
        for contour in contours_orange:
        
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
                shape = 'TRIANGLE'
                cv2.putText(img, 'TRIANGLE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                shape = 'SQUARE'
                cv2.putText(img, 'SQUARE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                shape = 'PENTAGON'
                cv2.putText(img, 'PENTAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                shape = 'HEXAGON'
                cv2.putText(img, 'HEXAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                shape = 'ROUND'
                cv2.putText(img, 'ROUND', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    elif colors[0] == 'TURQOISE':
        i = 0
        for contour in contours_turqoise:
        
            # here we are ignoring first counter because 
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue
        
            if cv2.contourArea(contour) < 3000:
                continue


            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True)
            
            # using drawContours() function
            cv2.drawContours(input_img, [contour], 0, (0, 0, 255), 5)
        
            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
        
            # putting shape name at center of each shape
            if len(approx) == 3:
                shape = 'TRIANGLE'
                cv2.putText(img, 'TRIANGLE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                shape = 'SQUARE'
                cv2.putText(img, 'SQUARE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                shape = 'PENTAGON'
                cv2.putText(img, 'PENTAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                shape = 'HEXAGON'
                cv2.putText(img, 'HEXAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                shape = 'ROUND'
                cv2.putText(img, 'ROUND', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif colors[0] == 'BLACK':
        i = 0
        for contour in contours_black:
        
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
                shape = 'TRIANGLE'
                cv2.putText(img, 'TRIANGLE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 4:
                shape = 'SQUARE'
                cv2.putText(img, 'SQUARE', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 5:
                shape = 'PENTAGON'
                cv2.putText(img, 'PENTAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            elif len(approx) == 6:
                shape = 'HEXAGON'
                cv2.putText(img, 'HEXAGON', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            else:
                shape = 'ROUND'
                cv2.putText(img, 'ROUND', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    
    detected = tci.detect_text(name)
    st.markdown(detected)

    descriptors = detected.split()
    descriptors = np.array(descriptors)

    

    pills= pd.read_csv("Pillbox.csv", low_memory = False)
    pills.drop(labels=['ID', 'Enabled?', 'created at', 'updated at', 
    'spp', 'setid', 'splsize', 'pillbox_size',
    'splscore', 'pillbox_score', 'pillbox_imprint', 
    'spl_strength', 'spl_ingredients', 
    'spl_inactive_ing', 'source', 'rxtty', 'rxstring', 'rxcui', 'RxNorm Update time', 
    'product_code', 'part_num', 'ndc9', 'ndc_labeler_code', 
    'ndc_product_code','splshape','splcolor', 'marketing_act_code', 'effective_time', 'file_name', 
    'equal_product_code', 'dosage_form', 'document_type', 'dea_schedule_code', 'dea_schedule_name', 
    'author_type', 'author', 'approval_code', 'image_source', 'splimage', 'has_image', 'epc_match',
    'version_number', 'pillbox_shape_text','pillbox_color_text','part_medicine_name','laberer_code', 'application_number', 'updated', 'stale', 'new', 'Pillbox Value']
            ,axis = 1, inplace= True)
    col = pills.pop('medicine_name')
    pills.set_index(col, inplace = True)

    pills.drop_duplicates()
    #detected = ["ORTH"]
    finalpill =[]
    #pills.loc[pills.splimprint.isin(detected)].index
    pills = (pills[(pills['splshape_text'] == shape) & (pills['splcolor_text'] == colors[0])])
    wowza = pills['splimprint'].str.contains(detected[0])
    wowza.dropna()
    display = wowza.sort_values(ascending = False)
    st.dataframe(display.head(5))
    print(wowza)
    
    for item in wowza:
        if item == True:
            finalpill.append(item)

    return finalpill

