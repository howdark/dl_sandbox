import pickle
import numpy as np
# from random import shuffle, seed, sample


with open('./data/creditcard.csv', 'r') as f:
    data = [line.replace('\"','').split(',') for line in f.read().splitlines()]
    data = data[1:]
    data = list(map(lambda x : [float(element) for element in x], data))


X, y = zip(*[(line[0:30], int(line[30])) for line in data])
y = list(y)

print('Number of samples Loaded')
print('0: ', len([d for d in y if d == 0]), '1: ', len([d for d in y if d == 1]))

X = np.array(X)
y = np.array(y)
normal_idx = [i for i, y in enumerate(y) if y == 0]
fraud_idx = [i for i, y in enumerate(y) if y == 1]
X_normal, X_fraud = X[normal_idx], X[fraud_idx]
y_normal, y_fraud = y[normal_idx], y[fraud_idx]

# ratio of test dataset
ratio = 0.2
# Randomly shuffle data
np.random.seed(10)

n_shuffled_idx = np.random.permutation(np.arange(len(y_normal)))
f_shuffled_idx = np.random.permutation(np.arange(len(y_fraud)))

X_normal_shuffled, y_normal_shuffled = X_normal[n_shuffled_idx], y_normal[n_shuffled_idx]
X_fraud_shuffled, y_fraud_shuffled = X_fraud[f_shuffled_idx], y_fraud[f_shuffled_idx]

n_test_sample_index = -1 * int(ratio * float(len(y_normal)))
f_test_sample_index = -1 * int(ratio * float(len(y_fraud)))

x_train = np.concatenate((X_normal_shuffled[:n_test_sample_index], X_fraud_shuffled[:f_test_sample_index]))
x_test = np.concatenate((X_normal_shuffled[n_test_sample_index:], X_fraud_shuffled[f_test_sample_index:]))
y_train = np.concatenate((y_normal_shuffled[:n_test_sample_index], y_fraud_shuffled[:f_test_sample_index]))
y_test = np.concatenate((y_normal_shuffled[n_test_sample_index:], y_fraud_shuffled[f_test_sample_index:]))

print(len(x_train), len(x_test), len(y_train), len(y_test))

# Randomly shuffle data
train_shuffled_idx = np.random.permutation(np.arange(len(y_train)))
test_shuffled_idx = np.random.permutation(np.arange(len(y_test)))

x_train_shuffled, y_train_shuffled = x_train[train_shuffled_idx], y_train[train_shuffled_idx]
x_test_shuffled, y_test_shuffled = x_test[test_shuffled_idx], y_test[test_shuffled_idx]

with open('./train_data.pkl', 'wb') as f:
    pickle.dump([x_train_shuffled, y_train_shuffled], f)

with open('./test_data.pkl', 'wb') as f:
    pickle.dump([x_test_shuffled, y_test_shuffled], f)

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


sm = SMOTE(kind='regular')
X_sm, y_sm = sm.fit_sample(x_train_shuffled, y_train_shuffled)
print('Samples oversampled to')
print('0: ', len([d for d in y_sm if d == 0]), '1: ', len([d for d in y_sm if d == 1]))

with open('./smote_data.pkl', 'wb') as f:
    pickle.dump([X_sm, y_sm], f)

