from __future__ import division
import numpy as np

def load_npz(file_name):
    arr = np.load(file_name)
    array = arr['arr_0']
    return array