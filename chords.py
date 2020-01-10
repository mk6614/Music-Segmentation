import statistics
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#chords represent a music time unit:
#a chord is a 4 beats for a 4/4 rhythm and 3 beats for 3/4 rhythm

def get_chords_part(from_, to, chords):
    return chords[:][...,from_:to]

def draw_cqt(C, save=False, storage='./', name='Chroma cqt', tlines=None, block=True):
    plt.rcParams['figure.figsize'] = (14, 5)
    plt.clf()
    librosa.display.specshow(C, cmap='gray_r', y_axis='chroma', x_axis='time')
    plt.colorbar(format='%+.2f')
    plt.tight_layout()
    plt.suptitle(name)
    if not tlines is None:
        plt.vlines(tlines, 0, 1000, alpha=0.8, color="r")
    if (save):
        plt.savefig(fname="{}{}_chords.pdf".format(storage, name))
        plt.clf()
        plt.close()
    else:
        plt.show(block = block)

#this beat extraction ony takes the background of the song and calculates beats on it
def extract_beat_samples(y,sr):
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full,aggregate=np.median,metric='cosine',width=int(librosa.time_to_frames(1, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i = 2
    power = 2
    mask_i = librosa.util.softmask(S_filter,margin_i * (S_full - S_filter),power=power)
    S_background = mask_i * S_full
    onset_env = librosa.onset.onset_strength(S=S_background)
    return librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units='samples')  

#chord variance measures the "clearness" of the chords; the lower the mean of each bar, the clearest the chord
def variation(chord_track):
    ret = []
    for j in range(0,len(chord_track[0])):
        ret.append(statistics.mean(list(chord_track[:,j])))
    return sum(ret)/len(ret)
    
def extract_chords(y, sr, beat_samples, rhythm, offset=0, draw=False, save=False):
    l = beat_samples[offset:].copy()
    l = l[::rhythm]
    y_samples = y[l[0]:l[-1]]

    l = l - beat_samples[offset]
    bar_t = librosa.samples_to_time(l, sr=sr)
    bar_f = librosa.samples_to_frames(l)
    
    bar_f = librosa.samples_to_frames(l)
    bar_t = librosa.samples_to_time(l, sr=sr)

    C = librosa.feature.chroma_cqt(y=y_samples, sr=sr)
    C_bar = librosa.util.sync(C, bar_f, aggregate=np.median)
    C_bar = librosa.util.normalize(C_bar)

    return C_bar
    
#clear chords are the chords thaat have minimum variance among beat offset from 0 to rhythm-1
def extract_clear_chords(y, sr, beat_samples, rhythm, draw=False, save=False):
    best = best_v = len(beat_samples)/rhythm
    for off in range(0,rhythm-1):
        C = extract_chords(y,sr,beat_samples,rhythm,offset=off,draw=draw,save=save)
        var = variation(C)
        print("sum of variance, offset {}: {}".format(off,var))
        if (var < best_v):
            best_v = var
            best = C
            b = off

    return best, b

#the key map function maps every chord to a number. It has the key at 0 and all othe chords at higher values; up to 11 for each of the 12 coresponding chords
def get_key_mapping(chord_track):
    func = len(chord_track) * [0] ##num of bins (12 chords) * [0]
    for j in range(0, len(chord_track[0])):
        func[(list(chord_track[:,j]).index(max(list(chord_track[:,j]))))] += 1
    #the chords that is played most throughout the song is named the key of the song
    max_ = func.index(max(func))
    tmp = list(range(len(chord_track)))
    return list(np.roll(tmp, max_)) #rolls the key to zero place; all other chords are higher that the key
    

#form key function, the chords are mapped to values between 0 to 11
def extract_function(chord_track, map_function):
    ret = []
    for i in range(0, len(chord_track[0])):
        ret.append(map_function[(list(chord_track[:,i]).index(max(list(chord_track[:,i]))))])
    return ret

	
