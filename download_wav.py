import urllib.request
import os
import time
import base64
import pydub

start = 'https://d34x6xks9kc6p2.cloudfront.net/'

location = 'sounds/'

ffmpeg_location = '/Users/coreyclemente/Documents/python_' + \
                  'projects/wav_project/lib/ffmpeg'

pydub.AudioSegment.converter = ffmpeg_location

#wavs = eval(open('combined_data.txt', 'r', encoding='utf8').read())

##def download():
##    to_find = [55, 61, 86, 112, 113, 115, 117, 126, 179, 182, \
##               190, 191, 192, 193, 250]
##    n = 0
##    s3_key = None
##    for i in to_find:
##            s3_key = wavs[i][1]['s3_key'][:-4] + '.mp3'
##            ID = wavs[i][0]['original_id'] + '.mp3'
##            print(start + s3_key)
##            urllib.request.urlretrieve(start + s3_key, location + ID)
##            print(str(n) + ' done!')
##            n += 1

def convert_all():
    directory = '/users/coreyclemente/Documents/sound_cube_analysis/sounds/'
    data = os.listdir(directory)[1:]
    for file in data:
        file = directory + file
        print(file)
        song = pydub.AudioSegment.from_file(file, 'mp3')
        song.export(file[:-4] + '.wav', format='wav')
        os.remove(file)
