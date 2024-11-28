from scipy.io import loadmat
import numpy as np

# initial
filepath = 'car1.mat'

def data_process():
    print('Data loading...')
    data = loadmat(filepath)['data']
    print(data.shape)
    print(data[0, 0])
