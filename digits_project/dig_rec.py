# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from matplotlib import pyplot as plt

# Load the classifier
model,pp = joblib.load("digits_cls1.pkl")

# Read the input image
im = cv2.imread("test1.png")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)
roi = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, np.ones((6,6), np.uint8))
plt.imshow(im_th,'gray')
plt.show()

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
labels= []
images = []


rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    # Draw the rectangles
    print(rect)
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    kernel = np.ones((int(leng*0.015), int(leng*0.015)), np.uint8)
    
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, kernel, iterations = 1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    #roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
##    plt.imshow(roi,'gray')
##    plt.show()
##    x= np.int32(input("num : "))
##    if(x >=0):
##        labels.append(x)
    # Calculate the HOG features
    roi_hog_fd = hog(roi,orientations= 10,
                     pixels_per_cell=(5, 5),
                     block_norm='L2-Hys',
                     cells_per_block=(1, 1),
                     visualize=False)
##    if(x>=0):
##        images.append(roi_hog_fd)
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
    # Predict digit
    nbr = model.predict(roi_hog_fd)
    print(nbr)
    cv2.putText(im, str(int(nbr)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

##model.partial_fit(images, labels)
##joblib.dump((model, pp), "digits_cls1.pkl", compress=3)
cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()

