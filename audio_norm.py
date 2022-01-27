import librosa
import os
import glob
import numpy as np
import scipy.io.wavfile as wavfile

data_path = '/home/ed716/Documents/NewSSD/LibriSpeech/train'
filenames = glob.glob(os.path.join(data_path, '*', '*', '*.wav'))

if(not os.path.isdir('libri_train_org')):
    os.mkdir('libri_train_org')

for file in filenames:
    name = file.split('/')[-1]
    name1 = '%04d'%(int(name.split('-')[0]))
    name2 = '%06d'%(int(name.split('-')[1]))
    name3 = '%04d'%(int(name.split('-')[2][:-4]))
    name = name1 + '_' + name2 + '_' + name3 + '.wav'
    if len(name) == 20:
        save_path = 'libri_train_org/' + name
        audio,_= librosa.load(file,sr=16000)
        max = np.max(np.abs(audio))
        norm_audio = np.divide(audio,max)
        print(save_path)
        wavfile.write(save_path, 16000, norm_audio)
