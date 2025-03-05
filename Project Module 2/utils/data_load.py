import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10(dir):
    test_data = unpickle('cifar-10-batches-py/test_batch')
    X_test = test_data[b'data']
    y_test = np.array(test_data[b'labels'])
    
    return X_test, y_test