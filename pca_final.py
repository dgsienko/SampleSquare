import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
import pyaudio
import wave

with open('pickled_data', 'rb') as read:
    all_data = pickle.load(read)
all_samples = all_data['matrix'].T
sample_ids = all_data['ids']

def PCA2D(X, ids):
    X = np.array(X)
    # Standardize
    X_std = StandardScaler().fit_transform(X)
    # PCA
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X_std)
    return combine(X_transf, ids)

def combine(X, ids, filename=None):
    all_data = []
    for i in range(len(X)):
        all_data += [[ids[i], X[i]]]
    if filename == None:
        return all_data
    else:
        f = open(filename, 'w')
        f.write(str(all_data))
        f.close()

def closest(X, point):
    smallest = [euclidean(point, X[0][1]), X[0]]
    distance = 0
    for points in X[1:]:
        distance = euclidean(point, points[1])
        if distance < smallest[0]: smallest = [distance, points[0]]
    return smallest

def play(name):
   #define stream chunk   
   chunk = 1024  
   #open a wav format music  
   f = wave.open(name)  
   #instantiate PyAudio  
   p = pyaudio.PyAudio()  
   #open stream  
   stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                   channels = f.getnchannels(),  
                   rate = f.getframerate(),  
                   output = True)  
   #read data  
   data = f.readframes(chunk)  
   #paly stream  
   while data != '':  
       stream.write(data)  
       data = f.readframes(chunk)  
   #stop stream  
   stream.stop_stream()  
   stream.close()  
   #close PyAudio  
   p.terminate()

def find_15(all_samples):
    all_samples = all_samples.T
    to_find = [55, 61, 86, 112, 113, 115, 117, 126, 179, 182, 190, 191, 192, 193, 250]
    data = []
    ids = []
    for i in to_find:
        data += [all_data['matrix'][i]]
        ids += [all_data['ids'][i]]
    return data, ids
