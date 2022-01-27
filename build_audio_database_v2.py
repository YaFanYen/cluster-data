import sys
sys.path.append("../../model/lib")
import os
import librosa
import numpy as np
import utils
import itertools
import time
import random
import scipy.io.wavfile as wavfile
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/libri_train'
norm_data = os.listdir(data_path)

# Parameter
SAMPLE_RANGE = (0,len(norm_data)) # data usage to generate database
WAV_REPO_PATH = os.path.expanduser("libri_train")
DATABASE_REPO_PATH = 'all_spk8'
NUM_SPEAKER = 10
MAX_NUM_SAMPLE = 5000

# time measure decorator
def timit(func):
    def cal_time(*args,**kwargs):
        tic = time.time()
        result = func(*args,**kwargs)
        tac = time.time()
        return result
    return cal_time

# create directory to store database
def init_dir(path = DATABASE_REPO_PATH ):
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isdir('%s/mix_wav'%path):
        os.mkdir('%s/mix_wav'%path)

@timit
def generate_path_list(sample_range=SAMPLE_RANGE,repo_path=WAV_REPO_PATH):
    '''
    :return: 2D array with idx and path (idx_wav,path_wav)
    '''
    audio_path_list = []

    for i in range(sample_range[0],sample_range[1]):
        path = repo_path + '/' + norm_data[i]
        if os.path.exists(path):
            audio_path_list.append((i,path))
    return audio_path_list

# data generate function
def single_audio_to_npy(audio_path_list,database_repo=DATABASE_REPO_PATH,fix_sr=8000):
    for idx,path in audio_path_list:
        data, _ = librosa.load(path, sr=fix_sr)
        data = utils.fast_stft(data)
        name = 'single-' + str(path[-29:-4])


# split single TF data to different part in order to mix
def split_to_mix(audio_path_list, database_repo=DATABASE_REPO_PATH, partition=2):
    # return split_list : (part1,part2,...)
    # each part : (idx,path)
    length = len(audio_path_list)
    part_len = length // partition
    head = 0
    part_idx = 0
    split_list = []
    while((head+part_len)<=length):
        part = audio_path_list[head:(head+part_len)]
        split_list.append(part)
        head = head + part_len
        part_idx = part_idx + 1
    return split_list

# mix single TF data
def all_mix(split_list,database_repo=DATABASE_REPO_PATH,partition=2):
    assert len(split_list) == partition
    print('mixing data...')
    num_mix = 1
    num_mix_check = 0
    for part in split_list:
        num_mix *= len(part)

    part_len = len(split_list[-1])
    idx_list = [x for x in range(part_len)]
    combo_idx_list = itertools.product(idx_list,repeat=partition)
    for combo_idx in combo_idx_list:
        num_mix_check +=1
        single_mix(combo_idx,split_list,database_repo)


# mix several wav file and store TF domain data with npy
def single_mix(combo_idx,split_list,database_repo):
    assert len(combo_idx) == len(split_list)
    mix_rate = 1.0 / float(len(split_list))
    wav_list = []
    prefix = 'mix-'
    mid_name = ''

    for part_idx in range(len(split_list)):
        random.shuffle(split_list[part_idx])
        idx,path = split_list[part_idx][combo_idx[part_idx]]
        wav, sr = librosa.load(path, sr=8000)
        wav_list.append(wav)
        name = path.split('/')[-1][:-4]
        mid_name += name + '+'
    wav_list = pad_sequences(wav_list, maxlen = 32000, dtype='float32', padding = 'post')

    # mix wav file
    mix_wav = np.zeros_like(wav_list[0])
    for wav in wav_list:
        mix_wav = mix_wav + wav# * mix_rate

    # save mix wav file
    wav_name = prefix + mid_name + '.wav'
    wav_name = wav_name.replace('+.wav', '.wav')
    s1 = wav_name[4:8]
    s2 = wav_name[21:25]
    s3 = wav_name[38:42]
    s4 = wav_name[55:59]
    s5 = wav_name[72:76]
    s6 = wav_name[89:93]
    s7 = wav_name[106:110]
    s8 = wav_name[123:127]
    s9 = wav_name[140:144]
    s10 = wav_name[157:161]
    if s1!=s2 and s1!=s3 and s1!=s4 and s1!=s5 and s1!=s6 and s1!=s7 and s1!=s8 and s2!=s3 and s2!=s4 and s2!=s5 and s2!=s6 and s2!=s7 and s2!=s8 and s3!=s4 and s3!=s5 and s3!=s6 and s3!=s7 and s3!=s8 and s4!=s5 and s4!=s6 and s4!=s7 and s4!=s8 and s5!=s6 and s5!=s7 and s5!=s8 and s6!=s7 and s6!=s8 and s7!=s8 and s1!=s9 and s1!=s10 and s2!=s9 and s2!=s10 and s3!=s9 and s3!=s10 and s4!=s9 and s4!=s10 and s5!=s9 and s5!=s10 and s6!=s9 and s6!=s10 and s7!=s9 and s7!=s10 and s8!=s9 and s8!=s10 and s9!=s10:
        wavfile.write('%s/%s'%(database_repo,wav_name),8000,mix_wav) #

if __name__ == "__main__":
    init_dir()
    audio_path_list = generate_path_list()
    single_audio_to_npy(audio_path_list)
    split_list = split_to_mix(audio_path_list,partition=NUM_SPEAKER)
    all_mix(split_list,partition=NUM_SPEAKER)
