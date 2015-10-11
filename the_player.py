from scipy.spatial.distance import euclidean
import pyaudio
import wave
import queue, threading

to_find = [55, 61, 86, 112, 113, 115, 117, 126, 179, 182, \
               190, 191, 192, 193, 250]

X = []

def play(name):  
   chunk = 1024   
   f = wave.open(name)  
   p = pyaudio.PyAudio()   
   stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                   channels = f.getnchannels(),  
                   rate = f.getframerate(),  
                   output = True)  
   data = f.readframes(chunk)  
   while data != '':  
       stream.write(data)  
       data = f.readframes(chunk)   
   stream.stop_stream()  
   stream.close()  
   p.terminate()

def player(point):
    id_to_play = chooser(X, point)
    play(id_to_play + '.mp3')

def chooser(X, point):
    smallest = [euclidean(point, X[0][1]), X[0]]
    distance = 0
    for points in X[1:]:
        distance = euclidean(point, points[1])
        if distance < smallest[0]: smallest = [distance, points[0]]
    return smallest
