from scipy.spatial.distance import euclidean
import pyaudio
import wave
import queue, threading
import numpy as np

class ThreadedTask(threading.Thread):
    def __init__(self, queue,task):
        threading.Thread.__init__(self)
        self.queue = queue
        self.task = task
    def run(self):
        try:
            self.task()
        except:
            pass

def process_queue(): # for init db
            try:
                msg = self.queue.get(0)
                stop_anim()
                toggle_disable()
            except queue.Empty:
                master.after(100, process_queue)


all_data = eval(open('combined_data.txt', 'r', encoding='utf8').read())

files = ['55e89f11e4b0b05b2dfad9ac', '55e89f1be4b0b05b2dfad9b0',\
         '55e29838e4b0b05b2dfad81c', '55f8bf17e4b0677f39924e8f', \
         '55fbb5aee4b0e7c5c6c95bf5','55fbb8ebe4b0e7c5c6c95c07', \
         '55fbb972e4b0e7c5c6c95c21','5610509ae4b0758498de5c11',\
         '55e360d0e4b0b05b2dfad840','55e0c89de4b0cfdb4696823d',\
         '55e74685e4b0b05b2dfad939','56058bf7e4b0758498de34f0',\
         '56058d16e4b0758498de34f4','56058d67e4b0758498de34f6',\
         '55fbbe7ce4b0e7c5c6c95c85']

X = np.array([[ 209.67319645,  141.93037207],
       [ 298.7223652 ,  300.        ],
       [  29.57460895,  203.88960193],
       [ 298.3266397 ,  138.54360527],
       [ 297.05010331,  140.15183977],
       [ 297.96181795,  139.49959215],
       [ 185.84702613,   30.11270426],
       [ 300.        ,  139.41536105],
       [ 226.90262985,    0.        ],
       [  48.50257606,  227.73585502],
       [ 107.49719208,   27.22637709],
       [   0.        ,  166.63025937],
       [  39.9588075 ,  111.32135093],
       [ 119.92787848,   24.23335298],
       [   6.49150816,  153.48486939]])

def play(name, thread):  
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
   thread.queue.put('Task finished')

def player(point, thread):
    i = chooser(X, point)
    file = '/users/coreyclemente/Documents/sound_cube_analysis/sounds/'
    name = file + files[i]
    play(name + '.wav', thread)

def chooser(X, point):
    smallest = [euclidean(point, (X[0][0], X[0][1])), 0]
    distance = 0
    for i in range(len(X[1:])):
        distance = euclidean(point, (X[i][0], X[i][1]))
        if distance < smallest[0]: smallest = [distance, i]
    return smallest[1]
