import keras
from keras.models import Sequential
from keras.layers import LSTM, Recurrent, Activation, Dense
from keras.optimizers import RMSprop


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

x_train = mnist.train.images.reshape((-1, 28, 28))
y_train = mnist.train.labels
x_test = mnist.test.images.reshape((-1, 28, 28))
y_test = mnist.test.labels

# Parameters
learning_rate = 0.001
training_iter = 100000
batch_size = 128
display_step = 10
epoch = 10

# Network Parameters
n_step = 28
n_input = 28
n_hidden = 128
n_classes = 10

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(n_step, n_input)))
model.add(Dense(10))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy', 'binary_accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1)

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\n')
print('test score : ', scores[0])
print('test accuracy : ', scores[1])
print('test binary_accuracy : ', scores[2])