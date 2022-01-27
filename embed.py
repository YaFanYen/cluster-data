import os
import os.path
import glob
import numpy as np
import random
import torch
import torchaudio
import librosa
import scipy.io.wavfile as wavfile
from dvector_data.wav2mel import Wav2Mel

wav2mel = Wav2Mel()
wav2mel = torch.jit.script(wav2mel)
wav2mel.save('wav2mel.pt')
dvector = torch.jit.load("dvector.pt").eval()
embed_audio_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/libri_embed'
libris = glob.glob(os.path.join(embed_audio_path, '*.wav'))

f = open('/home/ed716/Documents/NewSSD/Cocktail/audio/embedB.txt')
savepath = '/home/ed716/Documents/NewSSD/Cocktail/audio/2cluster/embedB'

embed = []
for line in f:
    embed.append(line)
f.close()

for i in range(len(embed)):
    idA = str(embed[i])
    sample_A1, sample_A2, sample_A3, sample_A4, sample_A5, sample_A6, sample_A7, sample_A8, sample_A9 = [], [], [], [], [], [], [], [], []
    if len(idA) == 6:
        idA1 = idA[:4]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
        mix_A = random.sample(sample_A1, 1)[0]

        wav_tensor, sample_rate = torchaudio.load(mix_A)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        save_name = os.path.join(savepath, idA.replace('\n', '.npy'))
        print(save_name)
        np.save(save_name, voice_speaker_emb)
    elif len(idA) == 11:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav = mix_wav1 + mix_wav2
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        print(save_name)
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
    elif len(idA) == 16:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        idA3 = idA[10:14]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
            if idA3 == id_name:
                sample_A3.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_A3 = random.sample(sample_A3, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav3, sr = librosa.load(mix_A3, sr=8000)
        mix_wav = mix_wav1 + mix_wav2 + mix_wav3
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
    elif len(idA) == 21:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        idA3 = idA[10:14]
        idA4 = idA[15:19]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
            if idA3 == id_name:
                sample_A3.append(libri)
            if idA4 == id_name:
                sample_A4.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_A3 = random.sample(sample_A3, 1)[0]
        mix_A4 = random.sample(sample_A4, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav3, sr = librosa.load(mix_A3, sr=8000)
        mix_wav4, sr = librosa.load(mix_A4, sr=8000)
        mix_wav = mix_wav1 + mix_wav2 + mix_wav3 + mix_wav4
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
    elif len(idA) == 26:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        idA3 = idA[10:14]
        idA4 = idA[15:19]
        idA5 = idA[20:24]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
            if idA3 == id_name:
                sample_A3.append(libri)
            if idA4 == id_name:
                sample_A4.append(libri)
            if idA5 == id_name:
                sample_A5.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_A3 = random.sample(sample_A3, 1)[0]
        mix_A4 = random.sample(sample_A4, 1)[0]
        mix_A5 = random.sample(sample_A5, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav3, sr = librosa.load(mix_A3, sr=8000)
        mix_wav4, sr = librosa.load(mix_A4, sr=8000)
        mix_wav5, sr = librosa.load(mix_A5, sr=8000)
        mix_wav = mix_wav1 + mix_wav2 + mix_wav3 + mix_wav4 + mix_wav5
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
        #'''#
    elif len(idA) == 31:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        idA3 = idA[10:14]
        idA4 = idA[15:19]
        idA5 = idA[20:24]
        idA6 = idA[25:29]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
            if idA3 == id_name:
                sample_A3.append(libri)
            if idA4 == id_name:
                sample_A4.append(libri)
            if idA5 == id_name:
                sample_A5.append(libri)
            if idA6 == id_name:
                sample_A6.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_A3 = random.sample(sample_A3, 1)[0]
        mix_A4 = random.sample(sample_A4, 1)[0]
        mix_A5 = random.sample(sample_A5, 1)[0]
        mix_A6 = random.sample(sample_A6, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav3, sr = librosa.load(mix_A3, sr=8000)
        mix_wav4, sr = librosa.load(mix_A4, sr=8000)
        mix_wav5, sr = librosa.load(mix_A5, sr=8000)
        mix_wav6, sr = librosa.load(mix_A6, sr=8000)
        mix_wav = mix_wav1 + mix_wav2 + mix_wav3 + mix_wav4 + mix_wav5 + mix_wav6
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
    elif len(idA) == 36:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        idA3 = idA[10:14]
        idA4 = idA[15:19]
        idA5 = idA[20:24]
        idA6 = idA[25:29]
        idA7 = idA[30:34]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
            if idA3 == id_name:
                sample_A3.append(libri)
            if idA4 == id_name:
                sample_A4.append(libri)
            if idA5 == id_name:
                sample_A5.append(libri)
            if idA6 == id_name:
                sample_A6.append(libri)
            if idA7 == id_name:
                sample_A7.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_A3 = random.sample(sample_A3, 1)[0]
        mix_A4 = random.sample(sample_A4, 1)[0]
        mix_A5 = random.sample(sample_A5, 1)[0]
        mix_A6 = random.sample(sample_A6, 1)[0]
        mix_A7 = random.sample(sample_A7, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav3, sr = librosa.load(mix_A3, sr=8000)
        mix_wav4, sr = librosa.load(mix_A4, sr=8000)
        mix_wav5, sr = librosa.load(mix_A5, sr=8000)
        mix_wav6, sr = librosa.load(mix_A6, sr=8000)
        mix_wav7, sr = librosa.load(mix_A7, sr=8000)
        mix_wav = mix_wav1 + mix_wav2 + mix_wav3 + mix_wav4 + mix_wav5 + mix_wav6 + mix_wav7
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
    elif len(idA) == 41:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        idA3 = idA[10:14]
        idA4 = idA[15:19]
        idA5 = idA[20:24]
        idA6 = idA[25:29]
        idA7 = idA[30:34]
        idA8 = idA[35:39]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
            if idA3 == id_name:
                sample_A3.append(libri)
            if idA4 == id_name:
                sample_A4.append(libri)
            if idA5 == id_name:
                sample_A5.append(libri)
            if idA6 == id_name:
                sample_A6.append(libri)
            if idA7 == id_name:
                sample_A7.append(libri)
            if idA8 == id_name:
                sample_A8.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_A3 = random.sample(sample_A3, 1)[0]
        mix_A4 = random.sample(sample_A4, 1)[0]
        mix_A5 = random.sample(sample_A5, 1)[0]
        mix_A6 = random.sample(sample_A6, 1)[0]
        mix_A7 = random.sample(sample_A7, 1)[0]
        mix_A8 = random.sample(sample_A8, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav3, sr = librosa.load(mix_A3, sr=8000)
        mix_wav4, sr = librosa.load(mix_A4, sr=8000)
        mix_wav5, sr = librosa.load(mix_A5, sr=8000)
        mix_wav6, sr = librosa.load(mix_A6, sr=8000)
        mix_wav7, sr = librosa.load(mix_A7, sr=8000)
        mix_wav8, sr = librosa.load(mix_A8, sr=8000)
        mix_wav = mix_wav1 + mix_wav2 + mix_wav3 + mix_wav4 + mix_wav5 + mix_wav6 + mix_wav7 + mix_wav8
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
    elif len(idA) == 46:
        idA1 = idA[:4]
        idA2 = idA[5:9]
        idA3 = idA[10:14]
        idA4 = idA[15:19]
        idA5 = idA[20:24]
        idA6 = idA[25:29]
        idA7 = idA[30:34]
        idA8 = idA[35:39]
        idA9 = idA[40:44]
        for libri in libris:
            id_name = str(libri.split('/')[-1][:-4])[:4]
            if idA1 == id_name:
                sample_A1.append(libri)
            if idA2 == id_name:
                sample_A2.append(libri)
            if idA3 == id_name:
                sample_A3.append(libri)
            if idA4 == id_name:
                sample_A4.append(libri)
            if idA5 == id_name:
                sample_A5.append(libri)
            if idA6 == id_name:
                sample_A6.append(libri)
            if idA7 == id_name:
                sample_A7.append(libri)
            if idA8 == id_name:
                sample_A8.append(libri)
            if idA9 == id_name:
                sample_A9.append(libri)
        mix_A1 = random.sample(sample_A1, 1)[0]
        mix_A2 = random.sample(sample_A2, 1)[0]
        mix_A3 = random.sample(sample_A3, 1)[0]
        mix_A4 = random.sample(sample_A4, 1)[0]
        mix_A5 = random.sample(sample_A5, 1)[0]
        mix_A6 = random.sample(sample_A6, 1)[0]
        mix_A7 = random.sample(sample_A7, 1)[0]
        mix_A8 = random.sample(sample_A8, 1)[0]
        mix_A9 = random.sample(sample_A9, 1)[0]
        mix_wav1, sr = librosa.load(mix_A1, sr=8000)
        mix_wav2, sr = librosa.load(mix_A2, sr=8000)
        mix_wav3, sr = librosa.load(mix_A3, sr=8000)
        mix_wav4, sr = librosa.load(mix_A4, sr=8000)
        mix_wav5, sr = librosa.load(mix_A5, sr=8000)
        mix_wav6, sr = librosa.load(mix_A6, sr=8000)
        mix_wav7, sr = librosa.load(mix_A7, sr=8000)
        mix_wav8, sr = librosa.load(mix_A8, sr=8000)
        mix_wav9, sr = librosa.load(mix_A9, sr=8000)
        mix_wav = mix_wav1 + mix_wav2 + mix_wav3 + mix_wav4 + mix_wav5 + mix_wav6 + mix_wav7 + mix_wav8 + mix_wav9
        save_name = os.path.join(savepath, idA.replace('\n', '') + '.wav')
        wavfile.write(save_name, 8000, mix_wav)

        wav_tensor, sample_rate = torchaudio.load(save_name)
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor)
        voice_speaker_emb = emb_tensor[np.newaxis, :].detach().numpy()      #(1, 256)
        np.save(save_name.replace('.wav', ''), voice_speaker_emb)
        command = 'sudo rm %s;'%(save_name)
        os.system(command)
#'''