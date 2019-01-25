import numpy as np 
import re 
import csv
import h5py
import pandas as pd

class Data(object):
    
    def __init__(self,
                 data_source,
                 alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 l0 = 1014,
                 batch_size = 128,
                 no_of_classes = 1):
        
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = no_of_classes
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1
        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source

    def loadData(self):
        pdFile = pd.read_hdf(self.data_source)
        self.data = pdFile.as_matrix()
        self.shuffled_data = self.data
        
    def shuffleData(self):
        np.random.seed(235)
        data_size = len(self.data)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]
        
    def getBatch(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = min((batch_num + 1) * self.batch_size, data_size)
        return self.shuffled_data[start_index:end_index]
        
    def getBatchToIndices(self, batch_num = 0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c)
            if c == 0:
                classes.append(np.zeros(1, dtype='int64'))
            else:
                classes.append(np.ones(1, dtype='int64'))
                                
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def getAllData(self):
        data_size = len(self.data)
        start_index = 0
        end_index = data_size 
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        classes = []
        batch_texts = self.data
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c)
            if c == 0:
                classes.append(np.zeros(1, dtype='int64'))
            else:
                classes.append(np.ones(1, dtype='int64'))
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)    
        
    def strToIndexs(self, s):
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        k = 0
        for i in range(1, n+1):
            c = s[-i]
            if c in self.dict:
                str2idx[i-1] = self.dict[c]
        return str2idx
    
    def getLength(self):
        return len(self.data)

if __name__ == '__main__':
    data = Data("data/generated_data/dataset.hdf5")
    saveTxt = True
    data.loadData()
    data.shuffleData()
    data.getAllData()
    with open("test.vec", "w") as fo:
        for i in range(data.getLength()):
            c = data.data[i][0]
            txt = data.data[i][1]
            vec = ",".join(map(str, data.strToIndexs(txt)))
            if saveTxt:
                vec = txt
            fo.write("{}\t{}\n".format(c, vec))