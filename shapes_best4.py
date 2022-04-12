import os
# https://pypi.org/project/opencv-python/
# pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
 
PATH = "...shapes\\"
IMG_SIZE = 64
shapes = ["circle", "square", "triangle", "star"]
labels = []
dataset = []
# https://www.kaggle.com/smeschke/four-shapes
# From kernel: https://www.kaggle.com/smeschke/load-data
for shape in shapes:
    print("Getting data for: ", shape)
    #iterate through each file in the folder
    for path in os.listdir(PATH + shape):
        #add the image to the list of images
        image = cv2.imread(PATH + shape + '/' + path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.reshape(12288)
        dataset.append(image)
        labels.append(shapes.index(shape))
 
    
#Look random images, five examples!
index = np.random.randint(0, len(dataset) - 1, size= 5)
plt.figure(figsize=(5,7))

 
for i, ind in enumerate(index, 1):
    img = dataset[ind].reshape((64, 64, 3))
    lab = shapes[labels[ind]]
    plt.subplot(4, 5, i)
    plt.title(lab)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    



    
X = np.array(dataset)
X.shape
 
y=np.array(labels)
y.shape
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
perceptron = Perceptron(max_iter=100, shuffle=True)
 
perceptron.fit(X_train, y_train)
perceptron.score(X_test, y_test)
print('score: ',perceptron.score(X_test, y_test))
 
y_pred = perceptron.predict(X_test)

bad_results = [(a,b,c) for (a,b,c) in zip(X_test[y_test != y_pred],
                                          y_test[y_test != y_pred],
                                          y_pred[y_test != y_pred] )]

print(f'number o errors in X_test: {len(bad_results)}')
#views errors:
i=1
for x_test, y_test, y_pred1 in bad_results:
    img = x_test.reshape((64, 64, 3))
    label_test = shapes[y_test]
    label_pred = shapes[y_pred1]
    plt.figure(figsize=(5,7))
    plt.subplot(len(bad_results), 1, i)
    plt.title(label_test +' - '+ label_pred)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    i+=1


y_pred = perceptron.predict(X)
perceptron.score(X,y)

#validation, random shape!
idx = randint(0,y_pred.size)
#idx = randint(0,2994)
print(idx)
plt.title(shapes[y_pred[idx]])
plt.imshow(X[idx].reshape((64,64,3)))
plt.show()

#----------------------------------

#check, predict  index number: ....


index = 12511
index_array = np.array([index])

#show what shape is for that index in dataset
plt.figure(figsize=(5,7))
for i, ind in enumerate(index_array, 1):
    img = dataset[ind].reshape((64, 64, 3))
    lab = shapes[labels[ind]]
    plt.subplot(4, 5, i)
    plt.title(lab)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


plt.title(shapes[y_pred[index]])
plt.imshow(X[index].reshape((64,64,3)))
plt.show()
