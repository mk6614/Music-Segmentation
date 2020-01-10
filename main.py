import librosa
import chords
import match

reference = "../original_lyrics.wav"
part_names =  ["intro", "verse 1", "chorus 1", "riff 1", "verse 2", "chorus 2", "riff 2", "outro"]
part_lengths = [8, 16, 10, 8, 16, 10, 16, 20] #in bars

remake_songs = ["../remake1.wav", "../remake2.wav", "../remake3.wav","../remake4.wav", "../remake5.wav"]
remake_names = ["Remake{}".format(i) for i in range(1,len(remake_songs)+1)]

rhythm = 4

def main():
	print("Please wait for the processing results: processing the reference song")
	ref_y,ref_sr = librosa.load(reference)
	tempo, ref_beat = chords.extract_beat_samples(ref_y, ref_sr)
	print("\testimated tempo: {}".format(tempo))
	ref_chords, ref_offset = chords.extract_clear_chords(ref_y, ref_sr, ref_beat, rhythm)
	ref_func = chords.extract_function(ref_chords, chords.get_key_mapping(ref_chords))
	print("\tFunction of the reference song:\n {}".format(ref_func))
	
	ref_part = match.find_representative_part(ref_func, part_lengths,draw=True, save=True, part_names=part_names)
	print("\trepresentative part: {}".format(ref_part))

	r_y = []
	r_sr = []
	r_beat = []
	r_offsets = []
	r_chords = []
	r_func = []
	match_points = []
	for i in range(len(remake_songs)):
		print("loading {}".format(remake_names[i]))
		y, sr = librosa.load(remake_songs[i])
		tempo, beat = chords.extract_beat_samples(y, sr)
		print("\testimated tempo: {}".format(tempo))
		chords_, offset = chords.extract_clear_chords(y, sr, beat, rhythm)
		func = chords.extract_function(chords_, chords.get_key_mapping(chords_))
		print("\t{} function:{}\n".format(remake_names[i], func))
		mp,_ = match.match_song(func,ref_part, draw=True, save=True, name=remake_names[i], structure=zip(part_lengths, part_names))
		print("\tmatch points: {}".format(mp))
		
		r_y.append(y)
		r_sr.append(sr)
		r_beat.append(beat)
		r_chords.append([chords_])
		r_offsets.append(offset)
		r_func.append(func)
		match_points.append(mp)
		
	return ref_part, ref_func, r_func, r_offsets, r_beat, r_y, r_sr, match_points

if __name__ == "__main__":
    main()