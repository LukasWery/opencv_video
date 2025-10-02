import cv2
import numpy as np

#find object function to find objects in an image using template matching
def findObject(Image, logo, treshold=0.8, label=""):
    Img= Image.copy()# prevent modification of original image
    # Convert to grayscale if necessary
    if len(Img.shape) == 3:
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    if len(logo.shape) == 3:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    #DO template matching and reform result to size of original image in heatmap
    result = cv2.matchTemplate(Img, logo, cv2.TM_CCOEFF_NORMED)
    heatmap = cv2.resize(result, (Img.shape[1], Img.shape[0])) #resize to original image size
    # Threshold to find matches 
    threshold = 0.8
    locs = np.where(result >= threshold)
    Lh, Lw = logo.shape[:2]  # Get height and width of the template
    # Draw rectangles around matches
    Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored rectangles
    for pt in zip(*locs[::-1]):
        cv2.rectangle(Img, pt, (pt[0] + Lw, pt[1] + Lh), (0, 0, 255), 2)
    #draw Image with rectangles and match heatmap
    cv2.imshow(f"{label} Original", Img)
    cv2.imshow(f"{label} Heatmap", heatmap)    
    return Img, heatmap #return original image and heatmap
    ############### 

#load images
img = cv2.imread("student_card.jpg", cv2.IMREAD_COLOR)
greyImg = cv2.imread("student_card.jpg", cv2.IMREAD_GRAYSCALE)
greyImg2 = greyImg.copy()
logo = cv2.imread('UT_logo.jpg', cv2.IMREAD_COLOR)
greyLogo = cv2.imread('UT_logo.jpg', cv2.IMREAD_GRAYSCALE)

#resize images
x=0.4
y=0.4
greyImg =cv2.resize(greyImg, (0, 0), fx=x, fy=y)
greyLogo = cv2.resize(greyLogo, (0, 0), fx=x, fy=y)


# Select ROI interactively PRESS ENTER OR SPACE TO CONFIRM SELECTION
roi_box = cv2.selectROI("Select Kernel Region", greyImg, showCrosshair=True)
x, y, w, h = roi_box
roi = greyImg[y:y+h, x:x+w]
cv2.destroyAllWindows()
findObject(greyImg, roi, treshold=0.2, label="ROI")


findObject(greyImg.copy(), greyLogo)


#cv2.imshow("logo", logo)
cv2.imshow("Student Card", greyImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
