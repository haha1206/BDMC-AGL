import numpy as np
import scipy.io as sio
import hdf5storage
def load_mat(path):

    data = sio.loadmat(path)

    X = np.squeeze(data['X'])

    Y = np.squeeze(data['Y'])


    return X,Y


