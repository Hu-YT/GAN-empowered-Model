from scipy.io import loadmat
import numpy as np
import torch
from sympy.codegen.ast import complex128


# initial
# filepath = 'car1.mat'

def data_process(filepath):
    print('Data loading...')
    data = loadmat(filepath)['data'].reshape(-1, 1)
    # empty_tensor = torch.zeros((72, 17, 10, 5), dtype=torch.complex128)
    # for i in range(72):
    #     for j in range(17):
    #         empty_tensor[i][j] = torch.from_numpy(data[i][j])
    empty_tensor = torch.zeros((1224, 10, 5), dtype=torch.float32)
    for i in range(1224):
            empty_tensor[i] = torch.from_numpy(data[i][0])
    data = empty_tensor
    return data