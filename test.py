# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy
import time
from ReadData import *
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import svm
from scipy import misc
from sklearn.preprocessing import normalize

trainImage = getTrainImage()
trainLabel = list(getTrainLabel())
#testImage = getTestImage()
#testLabel = list(getTestLabel())

X1 = []
#X2 = []
Y1 = trainLabel
#Y2 = testLabel

startTime = int(round(time.time()) * 1000)
for pixels in trainImage:
	image = []
	for i in range(0, 28):
		image.append(pixels[i * 28 : (i + 1) * 28])
	numpy.multiply(image, 255)	
	fd = hog(image, orientations=12,
                 pixels_per_cell=(4, 4),
                 block_norm='L2-Hys',
		 cells_per_block=(1, 1),
                 visualize=False)
	X1.append(fd)

clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(X1, Y1)
endTime = int(round(time.time()) * 1000)

#for pixels in testImage:
#	image = []
#	for i in range(0, 28):
#		image.append(pixels[i * 28 : (i + 1) * 28])
#	numpy.multiply(image, 255)	
#	fd = hog(image, orientations=12, pixels_per_cell=(2, 2), 
#					cells_per_block=(1, 1), visualize=False)
#	X2.append(fd)

#result = clf.predict(X2)

#errorCount = 0
#for i in range(0, len(result)):
#	if result[i] != Y2[i]:
#		errorCount += 1
#print('errorCount = ' + str(errorCount))
print('time elapse = ' + str(endTime - startTime))


image_index = 462
plt.imshow(trainImage[image_index].reshape(28,28),cmap='gray')
print(clf.predict( [X1[image_index]] ))
plt.show()


