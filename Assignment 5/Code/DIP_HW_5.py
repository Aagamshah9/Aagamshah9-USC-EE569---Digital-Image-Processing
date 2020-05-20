import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical

(i_train, o_train),(i_test, o_test) = cifar10.load_data()

# Creates Sequential Model
model = Sequential() 
# Creates 1st Convolutional Layer
model.add(Conv2D(6, kernel_size=(5,5), strides=(1, 1), padding = 'valid', activation = 'relu',kernel_initializer='glorot_normal', bias_initializer = 'random_uniform', input_shape = (32,32,3)))
# Creates 1st Max. Pooling Layer
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2), padding = 'valid'))
# Creates 2nd Convolutional Layer
model.add(Conv2D(16, kernel_size=(5,5), strides=(1, 1), padding = 'valid', activation = 'relu',kernel_initializer='glorot_normal', bias_initializer = 'random_uniform')) # We do not mention here the input size as it is mentioned only for the first input layer
# Creates 2nd Max. Pooling Layer
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2), padding = 'valid'))

# Creates Flatten Layer for transitioning into Fully Connected Layer
model.add(Flatten())
# Creates 120 Fully Connected Layers
model.add(Dense(120, activation = 'relu',kernel_initializer='glorot_normal', bias_initializer = 'random_uniform'))
model.add(Dropout(0.25))
# Creates 84 Fully Connected Layers
model.add(Dense(84, activation = 'relu',kernel_initializer='glorot_normal', bias_initializer = 'random_uniform'))
# Creates the Output Layer 
model.add(Dense(10, activation = 'softmax',kernel_initializer='glorot_normal',bias_initializer = 'random_uniform'))

# Compiling the Model created using Optimizers
# RMSprop
# model.compile(optimizer = optimizers.RMSprop(learning_rate = 0.001, decay=1e-6), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# SGD
#model.compile(optimizer = optimizers.SGD(learning_rate = 0.001, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Adagrad
#model.compile(optimizer = optimizers.Adagrad(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Adadelta
#model.compile(optimizer = optimizers.Adadelta(learning_rate = 1.0, rho=0.95), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Adam
model.compile(optimizer = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Adamax
#model.compile(optimizer = optimizers.Adamax(learning_rate = 0.002, beta_1 = 0.9, beta_2 = 0.999), loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Nadam
#model.compile(optimizer = optimizers.Nadam(learning_rate = 0.002, beta_1 = 0.9, beta_2 = 0.999), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training Model
history = model.fit(i_train/255, to_categorical(o_train), epochs = 100, batch_size = 128, validation_data = (i_test/255, to_categorical(o_test)))

# Testing Model
score = model.evaluate(i_test/255, to_categorical(o_test), batch_size = 128)

# Calculating the output results
print("Test Data Set Loss: %f" % score[0])
print("Test Data Set Accuracy: %f" % score[1])

# Plotting of Number of Epochs vs Accuracy

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()