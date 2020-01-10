import matplotlib.pyplot as plt
import statistics
import librosa
import scipy
import numpy as np
import chords
import librosa.display


#return the measure of similarity betwteen parts of length of ref_part
def part_similarity(part, ref_part):
    return sum([abs(ref_part[i]-part[i]) for i in range(len(ref_part))])


#corelets part with the whole song, taking the length of the part, and returns the differences for each offset and the best match
def corelate(part, whole):
    best = 100000
    offset = 0
    ln = len(part)
    diffs = []
    for i in range(0,len(whole) - ln+1):
        diff = part_similarity(whole[i:i+ln], part)
        diffs.append(diff)
        if (diff < best):
            best = diff
            offset = i
        
    return diffs, whole[offset:offset+ln]
    
#takes the calculated differences and the length of the part and return the points of good matches
def get_good_matches(diffs,part_len, draw=False, save=False, name="", structure=None):
    pd = 1
    padding = diffs[pd*-1:] + diffs
    #max_ = max(padding) + 3
    minimums ,_ = scipy.signal.find_peaks([-1*i for i in padding], distance = part_len)
    diffs_ = [padding[i]-pd for i in minimums]
    minimums = list(map(lambda x: x-pd,minimums))
    if draw:
        plt.ylabel("difference")
        plt.rcParams['figure.figsize'] = (14, 5)
        plt.clf()
        plt.plot(diffs)
        plt.xticks(minimums+ [len(diffs)+part_len])
        plt.title(name)
        plt.vlines(minimums, 0, diffs_, linestyles='dashdot', color='0.70')
        mean = statistics.mean(diffs_)
        plt.yticks([mean], ["(mean) " + str(mean)])
        plt.xlabel("offset (bars)")
        if not structure is None:
            i = 0
            tmp = 0.5
            for j, n in structure:
                if i > minimums[-1]:
                    break
                plt.hlines(-5, i, i+j)
                plt.vlines([i+j,i], -8, -2)
                plt.text(i+(j/2), -5, n, ha='center', va='bottom')
                i +=j
        if (save):
            plt.savefig(fname=name, format='pdf')
            plt.clf()
            plt.close()
        else:
            plt.show()

    return minimums, diffs_
    
#matches a remake song with the reference part:
#   the reference part is compared to the remake song to find the best matching sequence.
#   the sequence is again corelated with the remake itself.
#   The result are potential starting points of the song parts
def match_song(remake_func, ref_part_func, draw=False, save=False, name="", structure=None):
    matches, best_match = corelate(ref_part_func, remake_func)
    print("best match with the reference part: {}".format(best_match))
    diffs,_ = corelate(best_match, remake_func)
    return get_good_matches(diffs, len(best_match), draw=draw, save=save, name=name, structure=structure)


#structure is defined by sequential lengths of parts; measured in bars
def split_to_song_structure(structure, func):
    ret = []
    i = 0
    for j in structure:
        ret.append(func[i:i+j])
        i += j
    return ret
    
#correlates the parts of the reference song with the reference song itself.
#The most repetitive part is named the reference part
def find_representative_part(ref_func, part_lengths, draw=False, save=False, part_names=None):
    if part_names is None:
        part_names = list(range(part_lengths))
    parts = split_to_song_structure(part_lengths, ref_func)
    print("Song structure:")
    print([p for p in parts])
    min_ = 12 * len(ref_func)
    best_match = parts[0]
    for i,p in enumerate(parts):
        diffs, _ = corelate(p, ref_func)
        matches, diffs_ = get_good_matches(diffs, len(p),draw=draw, save=save,name="Reference song: part {}".format(part_names[i]), structure=zip(part_lengths,part_names))
        if not diffs_:
            continue
        match = statistics.mean(diffs_)
        if match < min_:
            min_ = match
            best_match = p

    return best_match



