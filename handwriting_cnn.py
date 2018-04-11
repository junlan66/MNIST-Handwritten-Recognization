# handwriting recognition using CNN
# Using python 2.7 to test, keras with tensorflow backend
import numpy
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.utils import np_utils
# will use mnist data set offered by keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 13
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train /255
X_test = X_test /255

# one hot encode the outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
# first dimension of image is 1 since gray scale images in mnist database
model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# using sigmoid activation
model.add(Conv2D(15, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
# adding softmax activation to last layer
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=7, batch_size=150)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy score: %.2f%%" % (scores[1]*100))

# serialize/save model to JSON
model_json = model.to_json()
with open("handwriting_model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved CNN in handwriting_model.json")