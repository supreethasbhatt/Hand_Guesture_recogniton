#import necessary packages
import numpy as np
import keras.models
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation, Dense
from dataset import data
from test import testdata
from testdataset import realtestdata 

#configure the neural network
np.random.seed(7)
model = Sequential()
model.add(Dense(1000, input_dim=4096))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


X = []
Y = []
count = 0

for d in data:
	count+=1
	im = np.array(Image.open(d))
	#im = Image.open(d).convert('L')
	#im = im.crop((500, 500, 3300 , 2500))
	#im.save(d)
	#im = np.array(im)
	#im.thumbnail((300,300), Image.ANTIALIAS)
	#im = im.convert('1')
	im = cv2.resize(im, (64,64))
	mat = np.array(im)
	mat1 = mat.flatten()
	X = np.concatenate((X, mat1), axis=0)
	Y.append(np.array(data[d]))

Y = np.array(Y)
Y = np.reshape(Y,(count,5))
X = np.array(X)
X = np.reshape(X, (366,4096))
history = model.fit(X, Y, batch_size=20, epochs=50)

X1=[]
Y1=[]
count=0
for d in testdata:
	count+=1
	im = Image.open(d)
	#print(im)
	#im = Image.open(d).convert('L')
	#im = im.crop((500, 900, 3300 , 2500))
	#im.save(d)
	im = np.array(im)
	im = cv2.resize(im, (64,64))
	mat = np.array(im)
	mat1 = mat.flatten()
	X1 = np.concatenate((X1, mat1), axis=0)
	Y1.append(np.array(testdata[d]))
	
Y1 = np.array(Y1)
Y1 = np.reshape(Y1,(count,5))
X1 = np.array(X1)
X1 = np.reshape(X1, (6, 4096))

score = model.evaluate(X1, Y1, batch_size=10)
#print(l)
print("\n\n Number of training images : 400 \n\n Number of testing images : 100 ")
print("\n\n The Testing accuracy is : {}".format(score[1] * 100))

print("\n Now evaluate with real data ..........")

X2=[]
Y2=[]
count=0
for d in realtestdata:
	count+=1
	im = Image.open(d)
	#print(im)
	#im = Image.open(d).convert('L')
	#im = im.crop((500, 900, 3300 , 2500))
	#im.save(d)
	im = np.array(im)
	im = cv2.resize(im, (64,64))
	mat = np.array(im)
	mat1 = mat.flatten()
	X2 = np.concatenate((X2, mat1), axis=0)
	Y2.append(np.array(testdata[d]))
	
Y2 = np.array(Y1)
Y2 = np.reshape(Y1,(count,5))
X2 = np.array(X1)
X2 = np.reshape(X1, (6, 4096))

score = model.evaluate(X2, Y2, batch_size=10)
#print(l)
print("\n\n Number of training images : 400 \n\n Number of testing images : 100 ")
print("\n\n The Testing accuracy is : {}".format(score[1] * 100))


