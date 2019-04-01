# Import the modules
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from ReadData import *
import time
from skimage.feature import hog
import pickle
from Training_System import *

#-------------------------------------------------------------------------------

def error_List(errorList):
    file = open("errorList.txt","w")
    for error in errorList:
        file.write(str(error).ljust(len(str(error))+1,"\n"))

def load_Image():
    # Load and split data
    trainImage = getTrainImage()
    trainLabel = list(getTrainLabel())
    testImage = getTestImage()
    testLabel = list(getTestLabel())
    
    X_train = []
    X_test = []
    Y_train = trainLabel
    Y_test = testLabel

    return trainImage, testImage, X_train, Y_train, X_test, Y_test

def hog_Image(Images):
    count = 0
    container = []
    for pixels in Images:
            image = []
            for i in range(0, 28):
                    image.append(pixels[i * 28 : (i + 1) * 28])
            np.multiply(image, 255)	
            fd = hog(image,orientations= 10,
                     pixels_per_cell=(5, 5),
                     block_norm='L2-Hys',
                     cells_per_block=(1, 1),
                     visualize=False)
            count += 1
            print("[","%.2f" %(count/len(Images)*100), "]", end="\r") 
            container.append(fd)
    print("\n","[DONE]","\n")
    hog_features = np.array(container, 'float64')
    return container
    
def train(model, images, labels):    
    startTime = int(round(time.time()))
    model.fit(images, labels)
    endTime = int(round(time.time()))
    print('time training = ' + str(endTime - startTime) + "s")
    return model

def train_more(model, images, labels):    
    startTime = int(round(time.time()))
    model.partial_fit(images, labels)
    endTime = int(round(time.time()))
    print('time training = ' + str(endTime - startTime) + "s")
    return model

def save(model):
    pkl_filename = "pickle_model.pkl"  
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(model, file)

def load():
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'rb') as file:  
        pickle_model = pickle.load(file)
    return pickle_model

#-------------------------------------------------------------------------------

trainImage, testImage, X_train, Y_train, X_test, Y_test = load_Image()

startTime = int(round(time.time()))

X_train = hog_Image(trainImage)
pp = preprocessing.StandardScaler().fit(X_train)
X_train = pp.transform(X_train)

X_test = hog_Image(testImage)
pp = preprocessing.StandardScaler().fit(X_test)
X_test = pp.transform(X_test)

endTime = int(round(time.time()))
print('time preprocessing = ' + str(endTime - startTime) + "s")

model = MLPClassifier(hidden_layer_sizes=(100, 3),
                      activation='logistic',
                      solver='adam',
                      learning_rate_init=0.001,
                      power_t=0.5,
                      max_iter=2000,
                      shuffle=True,
                      random_state=None,
                      tol=0.001,
                      verbose=False,
                      warm_start=True,
                      momentum=0.9,
                      nesterovs_momentum=True,
                      early_stopping=False,
                      validation_fraction=0.01,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      n_iter_no_change=10)
ans = input(">_")
if ans == "train":
    print("start training . . .")
    model = train(model, X_train, Y_train)
    save(model)
    joblib.dump((model, pp), "digits_cls1.pkl", compress=3)
elif ans == "load":
    model = load()
result = model.predict(X_test)

errorCount = 0
errorList = []
for i in range(0, len(result)):
	if result[i] != Y_test[i]:
		errorCount += 1
		errorList.append(i)
error_List(errorList)
print('errorCount = ' + str(errorCount))
score = model.score(X_train, Y_train)
print("Accuracy = {0:.2f}%".format(100 * score))
score = model.score(X_test, Y_test)
print("Accuracy = {0:.2f}%".format(100 * score))
#-------------------------------------------------------------------------------
re_images =[]
re_labels =[]
for error in errorList:
    re_images.append(X_test[error])
    re_labels.append(Y_test[error])
model = train_more(model, re_images, re_labels)
save(model)
joblib.dump((model, pp), "digits_cls1.pkl", compress=3)

#-------------------------------------------------------------------------------

while(True):
    image_index = int(input(">_"))
    if (image_index < 0 or image_index >= len(Y_test)):
        break
    plt.imshow(testImage[image_index].reshape(28,28),cmap='gray')
    x= model.predict([X_test[image_index]])
    print(x,(x == Y_test[image_index]))
    plt.show()
Predict(model, pp)
input("press any key to continue . . . ")
