import numpy as np
import scipy.special
import pandas as pd
#from mnist import MNIST as mnload
import keras    
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

np.random.seed(1)

###########################LOADING DATA###############
#load small dataset
data = pd.read_csv('mnist_train_100.csv', header=None)
x_train, y_train = data.iloc[:,1:], data.iloc[:,0]
x_train = (data / 255.0 * 0.99) + 0.01
numevents = x_train.shape[0]
#####################################################

################Neural Network Parameters#############
inputnodes = data.shape[1]      #input nodes from data
hiddennodes = 200               #number of hidden nodes
outputnodes = 10                #outputs node names
learningrate = 0.3              #learningrate
#######################################################

###########################NEURAL NETWORK CLASS#################################
class NeuralNetwork:
	
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		#set the nodes
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		#initiate weights
		self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
		#learning_rate
		self.lr = learningrate
		#activation_function
		self.activation_func = lambda x: scipy.special.expit(x)

	def train(self, inputs_list, targets_list):
		#convert list to array
		input = np.array(inputs_list, ndmin=2).T
		target = np.array(targets_list, ndmin=2).T
		#calculate signals
		hidden_input = np.dot(self.wih, input)
		hidden_output = self.activation_func(hidden_input)
		final_input = np.dot(self.who, hidden_output)
		final_output = self.activation_func(final_input)
		#calculate errors
		output_error = target - final_output
		hidden_error = np.dot(self.who.T, output_error)
		#update weights
		self.who += self.lr * np.dot((output_error * final_output * (1. - final_output)), np.transpose(hidden_output))
		self.wih += self.lr * np.dot((hidden_error * hidden_output *(1. - hidden_output)), np.transpose(input))

	def test(self, inputs_list):
		#convert inputs lists to 2d array
		input = np.array(inputs_list, ndmin=2).T
		#calculate signals into hidden layer
		hidden_input = np.dot(self.wih, input)
		hidden_output = self.activation_func(hidden_input)
		final_input = np.dot(self.who, hidden_output)
		final_output = self.activation_func(final_input)

		return final_output

######################################################

#######################################################
from PIL import Image, ImageFilter

def imageprepare(argv):
    im = Image.open(argv)
    # newImage.save("sample.png
    size = 28, 28
    im.thumbnail(size)
    imagear =  np.asarray(list(im.getdata()), dtype=np.float32)
    imagears = (0.2126 * imagear[:,0]) + (0.7152 * imagear[:,1]) + (0.0722 * imagear[:,2])
    imagears = 255 - imagears
    return imagears

######################################################

##################RUN NETWORK##########################
n = NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
for record in range(numevents):
	input = np.asarray(x_train.iloc[record])
	target = np.zeros(outputnodes) + 0.01
	target[y_train.iloc[record]] = 0.99
	n.train(input, target)
#######################################################

###################LOAD TEST DATA######################
test = pd.read_csv('mnist_test_10.csv', header=None)
x_test, y_test = test.iloc[:,1:], test.iloc[:,0]
x_test = (test / 255.0 * 0.99) + 0.01
#######################################################

###################RUN TEST###########################
num, match = 0, 0
while num<(test.shape[0]):
        if np.argmax(n.test(x_test.iloc[num,:])) == y_test[num]:
                match +=1
        num+=1
accuracy = (match/num)*100
print ('Accuracy = %d percent\n' %accuracy)
#######################################################


##########Comparison of performance with Keras#########
##################KERAS################################
input_shape = (28,28,1)
num_class = 10
num_epochs = 10

x_train, y_train = data.iloc[:,1:], data.iloc[:,0]
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28, 1))
y_train = keras.utils.to_categorical(y_train, num_class)
x_test, y_test = test.iloc[:,1:], test.iloc[:,0]
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28, 1))
y_test = keras.utils.to_categorical(y_test, num_class)
#######################################################

##################set up model#########################
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
#######################################################

###############Run and test############################
model.fit(x_train, y_train, epochs=num_epochs)
print (model.evaluate(x_test, y_test))
#######################################################
