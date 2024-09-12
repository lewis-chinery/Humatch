import math
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from Humatch.utils import seq_to_2D_kidera


class CustomDataGenerator(tf.keras.utils.Sequence):
    '''
    Custom generator is required for sparse data input to keras model
    tf.keras.utils.Sequence allows multiprocessing in safe way (won't train on same batch twice)

    :param seqs: list of aligned sequence strings
    :param batch_size: int, batch size for training
    '''
    def __init__(self, seqs, batch_size=1024):
        '''
        '''
        super().__init__()
        self.seqs = seqs
        self.batch_size = batch_size

    def __len__(self):
        '''
        Get the number of batches
        '''
        return math.ceil(len(self.seqs) / self.batch_size)

    def __getitem__(self, index):
        '''
        Get Kidera encoded ndarrays, X, for a batch of sequences
        '''
        low_idx = index*self.batch_size
        high_idx = min((index+1)*self.batch_size, len(self.seqs))
        batch_seqs = self.seqs[low_idx:high_idx]
        return get_X_from_list_of_seq_strs(batch_seqs)
    

def get_X_from_list_of_seq_strs(seq_strs):
    '''
    Get Kidera encoded ndarrays, X, for a list of sequences
    This array is required for CNN input

    :param seq_strs: list of str sequences
    :returns: ndarray of Kidera encoded (# seqs, 200, 10)
    '''
    with mp.Pool(mp.cpu_count()) as pool:
        X = np.asarray(pool.map(seq_to_2D_kidera, seq_strs))
    return X