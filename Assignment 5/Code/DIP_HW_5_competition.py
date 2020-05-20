import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Early Callback function 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

# Data Pre-processing
(data_train, label_train), (data_test, label_test) = cifar10.load_data()
data_train = data_train.astype('float32')
data_test = data_test.astype('float32') 
mean = np.mean(data_train,axis=(0,1,2,3))
std = np.std(data_train,axis=(0,1,2,3))
data_train = (data_train-mean)/(std+1e-7)
data_test = (data_test-mean)/(std+1e-7)
label_train = np_utils.to_categorical(label_train,10)
label_test = np_utils.to_categorical(label_test,10)

# Stratification of the data
fit_data_train, data_val, fit_label_train, label_val = train_test_split(data_train, label_train, test_size = 0.5, random_state = 0, stratify = label_train)
 
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), padding='same',stride = 1, kernel_regularizer=regularizers.l2(weight_decay), input_shape=data_train.shape[1:]))
# model.add(LeakyReLU(alpha=0.1))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = (3,3), padding='same',stride = 1, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(LeakyReLU(alpha=0.1))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, kernel_size = (3,3), padding='same',stride = 1, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(LeakyReLU(alpha=0.1))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = (3,3), padding='same',stride = 1, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(LeakyReLU(alpha=0.1))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, kernel_size = (3,3), padding='same',stride = 1, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(LeakyReLU(alpha=0.1))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = (3,3), padding='same',stride = 1, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(LeakyReLU(alpha=0.1))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
#model.add(Dense(units=256, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(Dropout(0.45))
#model.add(Dense(units=64, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(Dropout(0.45))
model.add(Dense(10, activation='softmax'))
 
model.summary()
 
# Data augmentation
datagenerator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagenerator.fit(data_train)
 
# Batch_size_for_training
batch_size = 64
 
opt_adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
history = model.fit_generator(datagenerator.flow(data_train, label_train, batch_size=batch_size),\
                    steps_per_epoch=data_train.shape[0] // batch_size,epochs=125,\
                    verbose=1,validation_data=(data_test,label_test),callbacks=[LearningRateScheduler(lr_schedule)])

score_train = model.evaluate(data_train, label_train)
print("Test Data Set Loss: %f" % score_train)

score_test = model.evaluate(data_test, label_test)
print("Test Data Set Accuracy: %f" % score_test)

# Plotting

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