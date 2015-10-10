def oneShot():
    f = open('one_shot_api.txt', encoding='utf8')
    s = f.read()
    return eval(s)

def spotifyData():
    f = open('spotify_data.txt', encoding='utf8')
    s = f.read()
    return eval(s)

def compare():
    both = []
    a = oneShot()
    b = spotifyData()
    for song in a:
        for data in b:
            if song['s3_key'] == data['s3_key']:
                both += [[song, data]]
                break
    f = open('combined_data.txt', 'w', encoding='utf8')
    f.write(str(both))
    f.close()
    print('Finished bitch!')
