import sys
sys.path.append("../../model/lib")
import os
import glob
import numpy as np
import shutil

data_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/libri_train'
save_path = '/home/ed716/Documents/NewSSD/Cocktail/audio/libri_embed/'
filenames = glob.glob(os.path.join(data_path, '*.wav'))

for filename in filenames:
    embed = filename.split('/')[-1][-8:-4]
    if embed == '0000':
        shutil.move(filename, save_path + filename.split('/')[-1])

