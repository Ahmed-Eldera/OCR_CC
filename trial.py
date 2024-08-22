import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils.object_detection import non_max_suppression
def show_image(image, title):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
def opening(image):
    kernel = np.ones((3, 3), np.uint8)

    # Apply opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Opened Image', opened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def closing(image):
    kernel = np.ones((3, 3), np.uint8)

    # Apply opening operation
    # Apply closing operation
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image
    # Display results
def tagroba(image):
    height,width=image.shape
    #imagez=image
    for y in range(height):
        for x in range(width):
            if image[y, x] >245:
                image[y,x] =255
            else:
                image[y,x]=0
def erosion(image, kernel_size=3, iterations=1):

    # Define the structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(image, kernel, iterations=iterations)

    return eroded_image
def dilation(image, kernel_size=3):
    """
    Apply dilation to a binary image using a specified structuring element.

    Parameters:
        image (numpy.ndarray): Input binary image (must be binary with 0 and 255 values).
        kernel_size (int): Size of the structuring element (must be odd).

    Returns:
        numpy.ndarray: Dilated image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create a 3x3 structuring element
    structuring_element = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation
    dilated_image = cv2.dilate(image, structuring_element, iterations=1)

    return dilated_image
def conV(img):
    nimg=img
    height, width = img.shape
    for y in range(2,height-2):
        for x in range(2,width-2):
            if img[y-1,x]==255  and img[y+1,x]==255 :
                nimg[y,x]=255
    return nimg
def conH(img):
    nimg=img
    height,width=img.shape
    for y in range(2,height-2):
        for x in range(2,width-2):
            if img[y,x-1]==255  and img[y,x+1]==255 :
                nimg[y,x]=255
    return nimg
def sharp(image, alpha=1):

    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + alpha, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Apply the sharpening filter
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image
def load_img():
    img = cv2.imread(r"C:\Users\hotoe\Desktop\01 - Straightforward.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Desktop\02 - You can do it.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Desktop\03 - Should be okay.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(r"C:\Users\hotoe\Desktop\05 - Looks cool, hope it runs cool too.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\06 - Hatetlewe7 hatlewe7.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Desktop\04 - Still ok, I hope.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\07 - Hatet3eweg hat3eweg.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\09 - El spero spathis we23et 3aaaa.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\10 - Mal7 w Felfel.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\11 - Ya setty ew3i.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\13 - Matozbot el camera ya Kimo.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\14 - 2el noor 2ata3.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(r"C:\Users\hotoe\Downloads\15 - Compresso Espresso.jpg", cv2.IMREAD_GRAYSCALE)
    return img
def padder(img,space=50):
    clr = int(img[5, 5])
    img = cv2.copyMakeBorder(
        img,
        space,
        space,
        space,
        space,
        cv2.BORDER_CONSTANT,
        value=[clr]  # Color of padding (black in this case)
    )
    ycord, xcord = img.shape
    for y in range(ycord):
        for x in range(xcord):
            if img[y, x] == clr:
                img[y, x] = 255  # White pixel or y- bottom<-10
    return img
def sobel(img):
    # @title Interactive Sobel+Canny { run: "auto", display-mode: "both" }
    ksizex = 3 # @param {type:"slider", min:1, max:13, step:2}
    ksizey = 1 # @param {type:"slider", min:1, max:13, step:2}
    scalex = 1 # @param {type:"slider", min:1, max:10, step:1}
    scaley = 2 # @param {type:"slider", min:1, max:10, step:1}
    deltax = 0 # @param {type:"slider", min:0, max:255, step:1}
    deltay = 0 # @param {type:"slider", min:0, max:255, step:1}
    threshold1 =200 # @param {type:"slider", min:1, max:255, step:1}
    threshold2 = 220 # @param {type:"slider", min:1, max:255, step:1}
    L2gradient = True # @param {type:"boolean"}
    dx = cv2.Sobel(img,cv2.CV_16S,1,0,None,ksizex,scalex,deltax)
    dy = cv2.Sobel(img,cv2.CV_16S,0,1,None,ksizey,scaley,deltay)
    sobelcanny = cv2.Canny(dx,dy,threshold1,threshold2,None,L2gradient)
    #show_image(sobelcanny,'5')
    return sobelcanny
def rotato(sobelcanny):
    edges = sobelcanny
    top = -1
    bottom = -1
    left = -1
    right = -1

    # Get image dimensions
    height, width = edges.shape

    # Find the highest white pixel in each row (for horizontal edge)
    for y in range(height):
        for x in range(width):
            if edges[y, x] == 255:  # White pixel or y- bottom<-10
                if top == -1:
                    top = y
                if y - bottom > 10:
                    bottom = y
                    bottomx = x
                if (left == -1 or x <= left):
                    left = x
                    lefty = y
                if right == -1 or x >= right:
                    right = x
                    righty = y
    print(f"Top: {top} , Bottom: {bottom},{bottomx} ,Left: {left},{lefty}, Right: {right}")
    distL = np.sqrt(pow((lefty - bottom), 2) + pow((left - bottomx), 2))
    distR = np.sqrt(pow((righty - bottom), 2) + pow((right - bottomx), 2))
    coef = 1

    if abs(bottomx - left) > 30 and abs(bottom - lefty) > 30:
        if distR > distL:
            lefty = righty
            left = right
            coef = -1
            print("help")
        print("rot")
        tangle = (bottom - lefty) / (bottomx - left)
        angle = np.degrees(np.arctan(tangle))
        w, h = img.shape
        print(angle)
        r = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        abs_cos = abs(r[0, 0])
        abs_sin = abs(r[0, 1])

        # Calculate the new width and height of the image
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to take into account the translation
        r[0, 2] += (new_w / 2) - w / 2
        r[1, 2] += (new_h / 2) - y / 2
        sobelcanny = cv2.warpAffine(sobelcanny, r, (new_w, new_h), flags=cv2.INTER_LINEAR)
        #show_image(sobelcanny, "aftr rot")
    return sobelcanny
# Function to load all template images from a folder
def load_templates(template_folder):
    templates = []
    for filename in os.listdir(template_folder):
        if filename.endswith('.png'):
            template_path = os.path.join(template_folder, filename)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            #template_img=padder(template_img,5)
            if template_img is not None:
                templates.append((filename, template_img))
    return templates

# Function to perform template matching and return the location of matches


def match_templates(target_img, templates, margin=5):
    matches = []

    for filename, template in templates:
        # Perform template matching
        result = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.62  # Adjust this threshold as needed

        # Find locations where the result exceeds the threshold
        locs = np.where(result >= threshold)
        locs = list(zip(*locs[::-1]))  # Convert to (x, y) coordinates

        # Get the size of the template
        template_height, template_width = template.shape

        # Create a list of bounding boxes and their probabilities
        boxes = []
        probabilities = []
        for (x, y) in locs:
            boxes.append((x, y, x + template_width, y + template_height))
            probabilities.append(result[y, x])

        # Apply non-maximum suppression to filter overlapping boxes
        boxes = np.array(boxes)
        probabilities = np.array(probabilities)

        if boxes.size > 0:
            pick = non_max_suppression(boxes, probs=probabilities, overlapThresh=0.3)

            # Filter out matches based on x position with a margin
            filtered_matches = []
            for (x1, y1, x2, y2) in pick:
                # Ensure coordinates are within bounds
                x1, y1 = int(max(x1, 0)), int(max(y1, 0))
                x2, y2 = int(min(x2, result.shape[1] - 1)), int(min(y2, result.shape[0] - 1))

                # Find the highest probability within this box
                submatrix = result[y1:y2, x1:x2]
                highest_prob = np.max(submatrix) if submatrix.size > 0 else 0

                # Check for overlap based on x position
                if not filtered_matches or abs(filtered_matches[-1][1][0] - x1) > margin:
                    filtered_matches.append((filename, (x1, y1), highest_prob))
                else:
                    # Update the existing match with the highest probability
                    prev_index = len(filtered_matches) - 1
                    if highest_prob > filtered_matches[prev_index][2]:
                        filtered_matches[prev_index] = (filename, (x1, y1), highest_prob)

            matches.extend(filtered_matches)

    return matches

# Function to print matched templates and their locations
def print_matched_templates(matches):
    sorted_matches = sorted(matches, key=lambda item: item[1][0])

    for filename, (x, y), _ in sorted_matches:
        print(filename)



    # Print matched templates and their locations
img = load_img()
img = padder(img)
#show_image(img,'')
sobelcanny =sobel(img)
sobelcanny = rotato(sobelcanny)


#corner detection and cropping
edges=sobelcanny
#plt.imshow(img, cmap='gray');
top = -1
bottom = -1
left = -1
right = -1

# Get image dimensions
height, width = edges.shape
for y in range(height):
    for x in range(width):
        if edges[y, x] > 100:  # White pixel
            if top == -1:
                top = y
            bottom = y
            bottomx = x
            if left == -1 or x < left:
                left = x
                lefty=y
            if right == -1 or x > right:
                right = x
                righty=y

# Print the found edge coordinates
print(f"Top: {top} , Bottom: {bottom},{bottomx} ,Left: {left},{lefty}, Right: {right}")
print(width,height)
# Ensure all boundaries are valid
if top != -1 and bottom != -1 and left != -1 and right != -1:
    # Crop the image using the bounding box
    cropped_img = sobelcanny[top:bottom+1, left:right+1]
    OG=img[top:bottom+1, left:right+1]
    #show_image(OG,'')
    cropped_img = cv2.resize(cropped_img, (1141,721), interpolation=cv2.INTER_AREA)
    height, width = cropped_img.shape
    cropped_img = cropped_img[int(0.5293*height):int(0.6643*height),int(0.0633*width):int(0.9219*width)]
    cropped_img = cv2.threshold(cropped_img, 50, 80, cv2.THRESH_BINARY)[1]
    # cropped_img = cv2.bitwise_not(cropped_img)
    cropped_img=closing(cropped_img)
    cropped_img=dilation(cropped_img,5)
    #cropped_img=opening(cropped_img)
    #cropped_img=erosion(cropped_img,3)
    cropped_img = cv2.bitwise_not(cropped_img)



    # Display the results
    templates = load_templates(r"C:\Users\hotoe\Desktop\templates")

    # Match templates
    matches = match_templates(cropped_img, templates)
    print_matched_templates(matches)
    show_image(cropped_img,'img')

else:
    print("No edges found!")

if img is None:
    print("Error loading image.")
else:
    #  plt.imshow(img_sobel2, cmap="gray")
    plt.show()











