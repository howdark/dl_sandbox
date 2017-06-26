import keras
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, SimpleRNN
from keras.optimizers import Adam, RMSprop

import pickle
import data_handle_keras_my as prep

# Parameters
# ==================================================

# Data loading params
dev_sample_percentage = .1  # "Percentage of the training data to use for validation"

# Model Hyperparameters
embedding_dim = 64
learning_rate = 0.001
hidden_unit = 64
dropout_keep_prob = 0.5     # "Dropout keep probability (default: 0.5)"
l2_reg_lambda = 0.0     # "L2 regularization lambda (default: 0.0)"

# Training parameters
batch_size = 128     # "Batch Size (default: 64)"
num_epochs = 10       # "Number of training epochs (default: 200)")

# Load data
"""
1. Loading data
2. Padding sentences
3. One-hot-encoding labels
"""
print("Loading Train data...")

# Train dataset
with open('train.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    train_docs, train_labels = pickle.load(f)

train_labels = prep.labeller(train_labels)

train_docs_p = prep.pad_sentences(train_docs)

print("Build vocabulary...")
# Build vocabulary
voc, voc_inv = prep.build_vocab(train_docs_p)
undefined_idx = voc['#UNDEFINED']
x, y = prep.build_input_data(train_docs_p, train_labels, voc)

print("Write vocabulary...")
# Write vocabulary
# Saving the objects:
with open('vocab.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([voc, voc_inv], f)

print("Loading Test data...")
# Test dataset
with open('test.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    test_docs, y_test = pickle.load(f)

y_test = prep.labeller(y_test)
# y_test = np.argmax(y_test, axis=1)
x_raw = prep.pad_sentences(test_docs)

with open('vocab.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    voc, voc_inv = pickle.load(f)

# Build vocabulary
x_test = prep.build_test_data(x_raw, voc, undefined_idx)


"""
Session Start
"""
print("Build model...")
model = Sequential()
model.add(Embedding(len(voc)+1, embedding_dim))
model.add(LSTM(hidden_unit, return_sequences=True))
model.add(LSTM(hidden_unit))
model.add(Dense(2, activation='softmax'))
# adam = Adam(lr=learning_rate)
rmsprop = RMSprop(lr=learning_rate)

print("Compile model...")
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

print("Fitting model...")
hist = model.fit(x, y, batch_size=batch_size, epochs=num_epochs, validation_split=.1, verbose=1)

print("Evaluate model...")
scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('\n')
print('test score : ', scores[0])
print('test accuracy : ', scores[1])
# print('test binary_accuracy : ', scores[2])
