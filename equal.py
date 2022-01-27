import sys
sys.path.append("../../model/lib")
import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/libri_train_org'
savepath = '/home/ed716/Documents/NewSSD/Cocktail/audio/libri_train'
org_data = os.listdir(data_path)

if(not os.path.isdir('libri_train')):
    os.mkdir('libri_train')

wav_list = []
for i in range(len(org_data)):
    path = data_path + '/' + org_data[i]
    wav, sr = librosa.load(path, sr=16000)   #(len(wav),)
    if wav.shape[0] < 48000:
        continue
    wav = wav[:48000, ]
    print(wav.shape)
    wav_list.append(wav)

for i in range(len(wav_list)):
    name = str(org_data[i])
    wavfile.write('%s/%s'%(savepath,name), 16000, wav_list[i])