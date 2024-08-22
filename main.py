import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import lab
for i in range(1,12):
    print(i)
    img = lab.load_img(i)
    trial = lab.padder(img)
    img = trial
    #lab.show_image(img,'')
    sobelcanny = lab.sobel(trial,40,130)
    trial = sobelcanny
    trial = lab.detect_and_draw_longest_lines(trial)
    clr = int(img[-5, -5])
    trial,OG = lab.process_image_for_horizontal_longest_line(trial,img,clr)
    #lab.show_image(OG,'')
    #lab.show_image(trial,'')
    SOG= lab.sobel(OG,40,130)

    lines = cv2.HoughLinesP(SOG, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    def help(state=False):
        def average_y_position(line):
            x1, y1, x2, y2 = line[0]
            return (y1 + y2) / 2

            # Sort lines based on the average y-position


        sorted_lines = sorted(lines, key=average_y_position,reverse=state)
        for i in sorted_lines:
            _,y1,_,y2 = i[0]
            #print(average_y_position(i))
        y,mx,mnx = lab.after_rot(sorted_lines)
        return y,mx,mnx
    def helpx(state=False):
        def average_y_position(line):
            x1, y1, x2, y2 = line[0]

            return (x1 + x2) / 2

            # Sort lines based on the average y-position


        sorted_lines = sorted(lines, key=average_y_position,reverse=state)
        for i in sorted_lines:
            _,y1,_,y2 = i[0]
            #print(average_y_position(i))
        y = lab.after_rotx(sorted_lines)
        return y
    #print(y)
    y, _, _ = help()
    OG = OG[:y,:]
    SOG= lab.sobel(OG,40,130)

    lines = cv2.HoughLinesP(SOG, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    y,mx,mnx=help(True)
    OG = OG[y:,:]
    SOG = lab.sobel(OG, 40, 130)
    lines = cv2.HoughLinesP(SOG, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    lines = lab.filter_and_convert_lines(lines)
    y=helpx(True)
    OG = OG[:, y:]
    width,height=OG.shape
    SOG = lab.sobel(OG, 40, 130)
    lines = cv2.HoughLinesP(SOG, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    lines = lab.filter_and_convert_lines(lines)
    y = helpx(True)
    OG = OG[:, y:]


    SOG = lab.sobel(OG, 40, 130)
    lines = cv2.HoughLinesP(SOG, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    #lines = lab.filter_and_convert_lines(lines)

    #OG = OG[:,:helpx()]
    lab.show_image(OG,str(i))



