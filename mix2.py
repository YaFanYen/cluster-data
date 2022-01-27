import os
import os.path
import glob
import numpy as np
import random
import torch
import torchaudio
import librosa
import scipy.io.wavfile as wavfile

libri_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/libri_train'
dataset_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/2cluster/mix1'
savepath = '/home/ed716/Documents/NewSSD/Cocktail/audio/2cluster/mix2'
savepath_A2 = '/home/ed716/Documents/NewSSD/Cocktail/audio/2cluster/A2'
f = open('/home/ed716/Documents/NewSSD/Cocktail/audio/mix1.txt')
fA = open('/home/ed716/Documents/NewSSD/Cocktail/audio/embedA.txt')
libris = glob.glob(os.path.join(libri_path, '*.wav'))

mix1 = []
embedA = []
for line in f:
    mix1.append(line)
f.close()
for line in fA:
    embedA.append(line)
fA.close()

for i in range(len(mix1)):
    print(mix1[i])
    mix1_path = os.path.join(dataset_path, mix1[i])
    B_path = mix1_path.replace('mix1/', 'B/').replace('.wav\n', 'spk2.wav')
    audio_B, sr = librosa.load(B_path, sr=8000)
    audio_mix1 = mix1[i].split('+')
    audio_A1, audio_A2, audio_A3, audio_A4, audio_A5, audio_A6, audio_A7, audio_A8, audio_A9, audio_A10 = str(audio_mix1[0])[4:], str(audio_mix1[1]), str(audio_mix1[2]), str(audio_mix1[3]), str(audio_mix1[4]), str(audio_mix1[5]), str(audio_mix1[6]), str(audio_mix1[7]), str(audio_mix1[8]), str(audio_mix1[9])[:-8]

    idA = str(embedA[i])
    idA1, idA2, idA3, idA4, idA5= '', '', '', '', ''
    sample_A1, sample_A2, sample_A3, sample_A4, sample_A5 = [], [], [], [], []
    if len(idA) == 6:
        idA1 += idA[:4]
        if idA1 == audio_A1[:4]:
            libri_A1 = audio_A1
        elif idA1 == audio_A2[:4]:
            libri_A1 = audio_A2
        elif idA1 == audio_A3[:4]:
            libri_A1 = audio_A3
        elif idA1 == audio_A4[:4]:
            libri_A1 = audio_A4
        elif idA1 == audio_A5[:4]:
            libri_A1 = audio_A5
        elif idA1 == audio_A6[:4]:
            libri_A1 = audio_A6
        elif idA1 == audio_A7[:4]:
            libri_A1 = audio_A7
        elif idA1 == audio_A8[:4]:
            libri_A1 = audio_A8
        elif idA1 == audio_A9[:4]:
            libri_A1 = audio_A9
        elif idA1 == audio_A10[:4]:
            libri_A1 = audio_A10
        for libri in libris:
            libri_name = str(libri.split('/')[-1][:-4])
            id_name = libri_name[:4]
            if (libri_A1 != libri_name) and (idA1 == id_name):
                sample_A1.append(libri)
        mix2_A1 = random.sample(sample_A1, 1)[0]
        name_A1 = mix2_A1.split('/')[-1][:-4]
        mix2_wav1, sr = librosa.load(mix2_A1, sr=8000)
        zero = np.zeros_like(mix2_wav1)
        mix2_wavA1 = zero + mix2_wav1
        mix2 = mix2_wav1 + audio_B
        name = mix1[i].replace('\n', '').replace(libri_A1, name_A1)
        wavfile.write('%s/%s'%(savepath, name), 8000, mix2)
        wavfile.write('%s/%s'%(savepath_A2, name.replace('.wav', 'spk1.wav')), 8000, mix2_wavA1)
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/mix2.txt','a') as f:
            f.write(name + '\n')
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/A2.txt','a') as f:
            f.write(name.replace('.wav', 'spk1.wav') + '\n')
    elif len(idA) == 11:
        idA1 += idA[:4]
        idA2 += idA[5:9]
        if idA1 == audio_A1[:4]:
            libri_A1 = audio_A1
        elif idA1 == audio_A2[:4]:
            libri_A1 = audio_A2
        elif idA1 == audio_A3[:4]:
            libri_A1 = audio_A3
        elif idA1 == audio_A4[:4]:
            libri_A1 = audio_A4
        elif idA1 == audio_A5[:4]:
            libri_A1 = audio_A5
        elif idA1 == audio_A6[:4]:
            libri_A1 = audio_A6
        elif idA1 == audio_A7[:4]:
            libri_A1 = audio_A7
        elif idA1 == audio_A8[:4]:
            libri_A1 = audio_A8
        elif idA1 == audio_A9[:4]:
            libri_A1 = audio_A9
        elif idA1 == audio_A10[:4]:
            libri_A1 = audio_A10

        if idA2 == audio_A1[:4]:
            libri_A2 = audio_A1
        elif idA2 == audio_A2[:4]:
            libri_A2 = audio_A2
        elif idA2 == audio_A3[:4]:
            libri_A2 = audio_A3
        elif idA2 == audio_A4[:4]:
            libri_A2 = audio_A4
        elif idA2 == audio_A5[:4]:
            libri_A2 = audio_A5
        elif idA2 == audio_A6[:4]:
            libri_A2 = audio_A6
        elif idA2 == audio_A7[:4]:
            libri_A2 = audio_A7
        elif idA2 == audio_A8[:4]:
            libri_A2 = audio_A8
        elif idA2 == audio_A9[:4]:
            libri_A2 = audio_A9
        elif idA2 == audio_A10[:4]:
            libri_A2 = audio_A10
        for libri in libris:
            libri_name = str(libri.split('/')[-1][:-4])
            id_name = libri_name[:4]
            if (libri_A1 != libri_name) and (idA1 == id_name):
                sample_A1.append(libri)
            if (libri_A2 != libri_name) and (idA2 == id_name):
                sample_A2.append(libri)
        mix2_A1 = random.sample(sample_A1, 1)[0]
        name_A1 = mix2_A1.split('/')[-1][:-4]
        mix2_A2 = random.sample(sample_A2, 1)[0]
        name_A2 = mix2_A2.split('/')[-1][:-4]
        mix2_wav1, sr = librosa.load(mix2_A1, sr=8000)
        mix2_wav2, sr = librosa.load(mix2_A2, sr=8000)
        mix2_wavA1 = mix2_wav1 + mix2_wav2
        mix2 = mix2_wav1 + mix2_wav2 + audio_B
        name = mix1[i].replace('\n', '').replace(libri_A1, name_A1).replace(libri_A2, name_A2)
        wavfile.write('%s/%s'%(savepath, name), 8000, mix2)
        wavfile.write('%s/%s'%(savepath_A2, name.replace('.wav', 'spk1.wav')), 8000, mix2_wavA1)
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/mix2.txt','a') as f:
            f.write(name + '\n')
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/A2.txt','a') as f:
            f.write(name.replace('.wav', 'spk1.wav') + '\n')
    elif len(idA) == 16:
        idA1 += idA[:4]
        idA2 += idA[5:9]
        idA3 += idA[10:14]
        if idA1 == audio_A1[:4]:
            libri_A1 = audio_A1
        elif idA1 == audio_A2[:4]:
            libri_A1 = audio_A2
        elif idA1 == audio_A3[:4]:
            libri_A1 = audio_A3
        elif idA1 == audio_A4[:4]:
            libri_A1 = audio_A4
        elif idA1 == audio_A5[:4]:
            libri_A1 = audio_A5
        elif idA1 == audio_A6[:4]:
            libri_A1 = audio_A6
        elif idA1 == audio_A7[:4]:
            libri_A1 = audio_A7
        elif idA1 == audio_A8[:4]:
            libri_A1 = audio_A8
        elif idA1 == audio_A9[:4]:
            libri_A1 = audio_A9
        elif idA1 == audio_A10[:4]:
            libri_A1 = audio_A10

        if idA2 == audio_A1[:4]:
            libri_A2 = audio_A1
        elif idA2 == audio_A2[:4]:
            libri_A2 = audio_A2
        elif idA2 == audio_A3[:4]:
            libri_A2 = audio_A3
        elif idA2 == audio_A4[:4]:
            libri_A2 = audio_A4
        elif idA2 == audio_A5[:4]:
            libri_A2 = audio_A5
        elif idA2 == audio_A6[:4]:
            libri_A2 = audio_A6
        elif idA2 == audio_A7[:4]:
            libri_A2 = audio_A7
        elif idA2 == audio_A8[:4]:
            libri_A2 = audio_A8
        elif idA2 == audio_A9[:4]:
            libri_A2 = audio_A9
        elif idA2 == audio_A10[:4]:
            libri_A2 = audio_A10

        if idA3 == audio_A1[:4]:
            libri_A3 = audio_A1
        elif idA3 == audio_A2[:4]:
            libri_A3 = audio_A2
        elif idA3 == audio_A3[:4]:
            libri_A3 = audio_A3
        elif idA3 == audio_A4[:4]:
            libri_A3 = audio_A4
        elif idA3 == audio_A5[:4]:
            libri_A3 = audio_A5
        elif idA3 == audio_A6[:4]:
            libri_A3 = audio_A6
        elif idA3 == audio_A7[:4]:
            libri_A3 = audio_A7
        elif idA3 == audio_A8[:4]:
            libri_A3 = audio_A8
        elif idA3 == audio_A9[:4]:
            libri_A3 = audio_A9
        elif idA3 == audio_A10[:4]:
            libri_A3 = audio_A10
        for libri in libris:
            libri_name = str(libri.split('/')[-1][:-4])
            id_name = libri_name[:4]
            if (libri_A1 != libri_name) and (idA1 == id_name):
                sample_A1.append(libri)
            if (libri_A2 != libri_name) and (idA2 == id_name):
                sample_A2.append(libri)
            if (libri_A3 != libri_name) and (idA3 == id_name):
                sample_A3.append(libri)
        mix2_A1 = random.sample(sample_A1, 1)[0]
        name_A1 = mix2_A1.split('/')[-1][:-4]
        mix2_A2 = random.sample(sample_A2, 1)[0]
        name_A2 = mix2_A2.split('/')[-1][:-4]
        mix2_A3 = random.sample(sample_A3, 1)[0]
        name_A3 = mix2_A3.split('/')[-1][:-4]
        mix2_wav1, sr = librosa.load(mix2_A1, sr=8000)
        mix2_wav2, sr = librosa.load(mix2_A2, sr=8000)
        mix2_wav3, sr = librosa.load(mix2_A3, sr=8000)
        mix2_wavA1 = mix2_wav1 + mix2_wav2 + mix2_wav3
        mix2 = mix2_wav1 + mix2_wav2 + mix2_wav3 + audio_B
        name = mix1[i].replace('\n', '').replace(libri_A1, name_A1).replace(libri_A2, name_A2).replace(libri_A3, name_A3)
        print('type', type(mix2_wavA1))
        wavfile.write('%s/%s'%(savepath, name), 8000, mix2)
        wavfile.write('%s/%s'%(savepath_A2, name.replace('.wav', 'spk1.wav')), 8000, mix2_wavA1)
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/mix2.txt','a') as f:
            f.write(name + '\n')
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/A2.txt','a') as f:
            f.write(name.replace('.wav', 'spk1.wav') + '\n')
    elif len(idA) == 21:
        idA1 += idA[:4]
        idA2 += idA[5:9]
        idA3 += idA[10:14]
        idA4 += idA[15:19]
        if idA1 == audio_A1[:4]:
            libri_A1 = audio_A1
        elif idA1 == audio_A2[:4]:
            libri_A1 = audio_A2
        elif idA1 == audio_A3[:4]:
            libri_A1 = audio_A3
        elif idA1 == audio_A4[:4]:
            libri_A1 = audio_A4
        elif idA1 == audio_A5[:4]:
            libri_A1 = audio_A5
        elif idA1 == audio_A6[:4]:
            libri_A1 = audio_A6
        elif idA1 == audio_A7[:4]:
            libri_A1 = audio_A7
        elif idA1 == audio_A8[:4]:
            libri_A1 = audio_A8
        elif idA1 == audio_A9[:4]:
            libri_A1 = audio_A9
        elif idA1 == audio_A10[:4]:
            libri_A1 = audio_A10
        if idA2 == audio_A1[:4]:
            libri_A2 = audio_A1
        elif idA2 == audio_A2[:4]:
            libri_A2 = audio_A2
        elif idA2 == audio_A3[:4]:
            libri_A2 = audio_A3
        elif idA2 == audio_A4[:4]:
            libri_A2 = audio_A4
        elif idA2 == audio_A5[:4]:
            libri_A2 = audio_A5
        elif idA2 == audio_A6[:4]:
            libri_A2 = audio_A6
        elif idA2 == audio_A7[:4]:
            libri_A2 = audio_A7
        elif idA2 == audio_A8[:4]:
            libri_A2 = audio_A8
        elif idA2 == audio_A9[:4]:
            libri_A2 = audio_A9
        elif idA2 == audio_A10[:4]:
            libri_A2 = audio_A10

        if idA3 == audio_A1[:4]:
            libri_A3 = audio_A1
        elif idA3 == audio_A2[:4]:
            libri_A3 = audio_A2
        elif idA3 == audio_A3[:4]:
            libri_A3 = audio_A3
        elif idA3 == audio_A4[:4]:
            libri_A3 = audio_A4
        elif idA3 == audio_A5[:4]:
            libri_A3 = audio_A5
        elif idA3 == audio_A6[:4]:
            libri_A3 = audio_A6
        elif idA3 == audio_A7[:4]:
            libri_A3 = audio_A7
        elif idA3 == audio_A8[:4]:
            libri_A3 = audio_A8
        elif idA3 == audio_A9[:4]:
            libri_A3 = audio_A9
        elif idA3 == audio_A10[:4]:
            libri_A3 = audio_A10

        if idA4 == audio_A1[:4]:
            libri_A4 = audio_A1
        elif idA4 == audio_A2[:4]:
            libri_A4 = audio_A2
        elif idA4 == audio_A3[:4]:
            libri_A4 = audio_A3
        elif idA4 == audio_A4[:4]:
            libri_A4 = audio_A4
        elif idA4 == audio_A5[:4]:
            libri_A4 = audio_A5
        elif idA4 == audio_A6[:4]:
            libri_A4 = audio_A6
        elif idA4 == audio_A7[:4]:
            libri_A4 = audio_A7
        elif idA4 == audio_A8[:4]:
            libri_A4 = audio_A8
        elif idA4 == audio_A9[:4]:
            libri_A4 = audio_A9
        elif idA4 == audio_A10[:4]:
            libri_A4 = audio_A10
        for libri in libris:
            libri_name = str(libri.split('/')[-1][:-4])
            id_name = libri_name[:4]
            if (libri_A1 != libri_name) and (idA1 == id_name):
                sample_A1.append(libri)
            if (libri_A2 != libri_name) and (idA2 == id_name):
                sample_A2.append(libri)
            if (libri_A3 != libri_name) and (idA3 == id_name):
                sample_A3.append(libri)
            if (libri_A4 != libri_name) and (idA4 == id_name):
                sample_A4.append(libri)
        mix2_A1 = random.sample(sample_A1, 1)[0]
        name_A1 = mix2_A1.split('/')[-1][:-4]
        mix2_A2 = random.sample(sample_A2, 1)[0]
        name_A2 = mix2_A2.split('/')[-1][:-4]
        mix2_A3 = random.sample(sample_A3, 1)[0]
        name_A3 = mix2_A3.split('/')[-1][:-4]
        mix2_A4 = random.sample(sample_A4, 1)[0]
        name_A4 = mix2_A4.split('/')[-1][:-4]
        mix2_wav1, sr = librosa.load(mix2_A1, sr=8000)
        mix2_wav2, sr = librosa.load(mix2_A2, sr=8000)
        mix2_wav3, sr = librosa.load(mix2_A3, sr=8000)
        mix2_wav4, sr = librosa.load(mix2_A4, sr=8000)
        mix2_wavA1 = mix2_wav1 + mix2_wav2 + mix2_wav3 + mix2_wav4
        mix2 = mix2_wav1 + mix2_wav2 + mix2_wav3 + mix2_wav4 + audio_B
        name = mix1[i].replace('\n', '').replace(libri_A1, name_A1).replace(libri_A2, name_A2).replace(libri_A3, name_A3).replace(libri_A4, name_A4)
        wavfile.write('%s/%s'%(savepath, name), 8000, mix2)
        wavfile.write('%s/%s'%(savepath_A2, name.replace('.wav', 'spk1.wav')), 8000, mix2_wavA1)
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/mix2.txt','a') as f:
            f.write(name + '\n')
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/A2.txt','a') as f:
            f.write(name.replace('.wav', 'spk1.wav') + '\n')
    elif len(idA) == 26:
        idA1 += idA[:4]
        idA2 += idA[5:9]
        idA3 += idA[10:14]
        idA4 += idA[15:19]
        idA5 += idA[20:24]
        if idA1 == audio_A1[:4]:
            libri_A1 = audio_A1
        elif idA1 == audio_A2[:4]:
            libri_A1 = audio_A2
        elif idA1 == audio_A3[:4]:
            libri_A1 = audio_A3
        elif idA1 == audio_A4[:4]:
            libri_A1 = audio_A4
        elif idA1 == audio_A5[:4]:
            libri_A1 = audio_A5
        elif idA1 == audio_A6[:4]:
            libri_A1 = audio_A6
        elif idA1 == audio_A7[:4]:
            libri_A1 = audio_A7
        elif idA1 == audio_A8[:4]:
            libri_A1 = audio_A8
        elif idA1 == audio_A9[:4]:
            libri_A1 = audio_A9
        elif idA1 == audio_A10[:4]:
            libri_A1 = audio_A10
            
        if idA2 == audio_A1[:4]:
            libri_A2 = audio_A1
        elif idA2 == audio_A2[:4]:
            libri_A2 = audio_A2
        elif idA2 == audio_A3[:4]:
            libri_A2 = audio_A3
        elif idA2 == audio_A4[:4]:
            libri_A2 = audio_A4
        elif idA2 == audio_A5[:4]:
            libri_A2 = audio_A5
        elif idA2 == audio_A6[:4]:
            libri_A2 = audio_A6
        elif idA2 == audio_A7[:4]:
            libri_A2 = audio_A7
        elif idA2 == audio_A8[:4]:
            libri_A2 = audio_A8
        elif idA2 == audio_A9[:4]:
            libri_A2 = audio_A9
        elif idA2 == audio_A10[:4]:
            libri_A2 = audio_A10

        if idA3 == audio_A1[:4]:
            libri_A3 = audio_A1
        elif idA3 == audio_A2[:4]:
            libri_A3 = audio_A2
        elif idA3 == audio_A3[:4]:
            libri_A3 = audio_A3
        elif idA3 == audio_A4[:4]:
            libri_A3 = audio_A4
        elif idA3 == audio_A5[:4]:
            libri_A3 = audio_A5
        elif idA3 == audio_A6[:4]:
            libri_A3 = audio_A6
        elif idA3 == audio_A7[:4]:
            libri_A3 = audio_A7
        elif idA3 == audio_A8[:4]:
            libri_A3 = audio_A8
        elif idA3 == audio_A9[:4]:
            libri_A3 = audio_A9
        elif idA3 == audio_A10[:4]:
            libri_A3 = audio_A10

        if idA4 == audio_A1[:4]:
            libri_A4 = audio_A1
        elif idA4 == audio_A2[:4]:
            libri_A4 = audio_A2
        elif idA4 == audio_A3[:4]:
            libri_A4 = audio_A3
        elif idA4 == audio_A4[:4]:
            libri_A4 = audio_A4
        elif idA4 == audio_A5[:4]:
            libri_A4 = audio_A5
        elif idA4 == audio_A6[:4]:
            libri_A4 = audio_A6
        elif idA4 == audio_A7[:4]:
            libri_A4 = audio_A7
        elif idA4 == audio_A8[:4]:
            libri_A4 = audio_A8
        elif idA4 == audio_A9[:4]:
            libri_A4 = audio_A9
        elif idA4 == audio_A10[:4]:
            libri_A4 = audio_A10

        if idA5 == audio_A1[:4]:
            libri_A5 = audio_A1
        elif idA5 == audio_A2[:4]:
            libri_A5 = audio_A2
        elif idA5 == audio_A3[:4]:
            libri_A5 = audio_A3
        elif idA5 == audio_A4[:4]:
            libri_A5 = audio_A4
        elif idA5 == audio_A5[:4]:
            libri_A5 = audio_A5
        elif idA5 == audio_A6[:4]:
            libri_A5 = audio_A6
        elif idA5 == audio_A7[:4]:
            libri_A5 = audio_A7
        elif idA5 == audio_A8[:4]:
            libri_A5 = audio_A8
        elif idA5 == audio_A9[:4]:
            libri_A5 = audio_A9
        elif idA5 == audio_A10[:4]:
            libri_A5 = audio_A10
        for libri in libris:
            libri_name = str(libri.split('/')[-1][:-4])
            id_name = libri_name[:4]
            if (libri_A1 != libri_name) and (idA1 == id_name):
                sample_A1.append(libri)
            if (libri_A2 != libri_name) and (idA2 == id_name):
                sample_A2.append(libri)
            if (libri_A3 != libri_name) and (idA3 == id_name):
                sample_A3.append(libri)
            if (libri_A4 != libri_name) and (idA4 == id_name):
                sample_A4.append(libri)
            if (libri_A5 != libri_name) and (idA5 == id_name):
                sample_A5.append(libri)
        mix2_A1 = random.sample(sample_A1, 1)[0]
        name_A1 = mix2_A1.split('/')[-1][:-4]
        mix2_A2 = random.sample(sample_A2, 1)[0]
        name_A2 = mix2_A2.split('/')[-1][:-4]
        mix2_A3 = random.sample(sample_A3, 1)[0]
        name_A3 = mix2_A3.split('/')[-1][:-4]
        mix2_A4 = random.sample(sample_A4, 1)[0]
        name_A4 = mix2_A4.split('/')[-1][:-4]
        mix2_A5 = random.sample(sample_A5, 1)[0]
        name_A5 = mix2_A5.split('/')[-1][:-4]
        mix2_wav1, sr = librosa.load(mix2_A1, sr=8000)
        mix2_wav2, sr = librosa.load(mix2_A2, sr=8000)
        mix2_wav3, sr = librosa.load(mix2_A3, sr=8000)
        mix2_wav4, sr = librosa.load(mix2_A4, sr=8000)
        mix2_wav5, sr = librosa.load(mix2_A5, sr=8000)
        mix2_wavA1 = mix2_wav1 + mix2_wav2 + mix2_wav3 + mix2_wav4 + mix2_wav5
        mix2 = mix2_wav1 + mix2_wav2 + mix2_wav3 + mix2_wav4 + mix2_wav5 + audio_B
        name = mix1[i].replace('\n', '').replace(libri_A1, name_A1).replace(libri_A2, name_A2).replace(libri_A3, name_A3).replace(libri_A4, name_A4).replace(libri_A5, name_A5)
        wavfile.write('%s/%s'%(savepath, name), 8000, mix2)
        wavfile.write('%s/%s'%(savepath_A2, name.replace('.wav', 'spk1.wav')), 8000, mix2_wavA1)
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/mix2.txt','a') as f:
            f.write(name + '\n')
        with open('/home/ed716/Documents/NewSSD/Cocktail/audio/A2.txt','a') as f:
            f.write(name.replace('.wav', 'spk1.wav') + '\n')

