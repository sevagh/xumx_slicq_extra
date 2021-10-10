import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import glob
from essentia.standard import EasyLoader
import essentia.pytools.spectral as sp
from sliceq22.overlap import nsgcq_overlap_add
from sliceq22.audio import SNR, extract_segment_randomly


# copied from https://raw.githubusercontent.com/f90/Wave-U-Net-Pytorch/master/data/musdb.py
def _get_musdbhq(database_path):
    '''
    Retrieve audio file paths for MUSDB HQ dataset
    :param database_path: MUSDB HQ root directory
    :return: dictionary with train and test keys, each containing list of samples, each sample containing all audio paths
    '''
    subsets = list()

    for subset in ["train", "test"]:
        print("Loading " + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = list()

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            example = dict()
            for stem in ["mix", "bass", "drums", "other", "vocals"]:
                filename = stem if stem != "mix" else "mixture"
                audio_path = os.path.join(track_folder, filename + ".wav")
                example[stem] = audio_path

            samples.append(example)
        subsets.append(samples)
    return subsets


# copied from https://raw.githubusercontent.com/f90/Wave-U-Net-Pytorch/master/data/musdb.py
def get_musdbhq(root_path):
    dataset = _get_musdbhq(root_path)

    train_val_list = dataset[0]
    test_list = dataset[1]

    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    # print("First training song: " + str(train_list[0])) # To debug whether partitioning is deterministic
    return {"train" : train_list, "val" : val_list, "test" : test_list}


class MUSDBDataGenerator(keras.utils.Sequence):
    def __init__(self, musdb_songs, nsg_params, seq_len=10.0, sr=44100, batch_size=32, shuffle=True):
        self.musdb_songs = musdb_songs
        self.batch_size = batch_size
        self.sr = sr
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.nsg_params = nsg_params
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.musdb_songs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        songs_tmp = [self.musdb_songs[k] for k in indices]
        x, y = self.generate_data(songs_tmp)
        return x, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.musdb_songs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def generate_data(self, song_subset):
        xs = []
        ys = []
        for song_dict in song_subset:
            song = song_dict['mix'] # use mixed audio from MUSDB18-HQ only, for now

            x_full = EasyLoader(filename=song, sampleRate=self.sr)()
            x = extract_segment_randomly(x_full, self.sr, self.seq_len)

            # Forward and backward transforms
            cq_frames, dc_frames, nb_frames = sp.nsgcqgram(x, **self.nsg_params)

            n_cq_frames = len(cq_frames)
            cq_time_coefs = cq_frames[0].shape[-1]
            total_coefs = n_cq_frames * cq_time_coefs

            cq_frames_ndarray = np.asarray(cq_frames)
            mag_cq = np.abs(cq_frames_ndarray)

            mag_cq_ola = nsgcq_overlap_add(mag_cq)

            # input is the overlap-added slicq
            xs.append(tf.convert_to_tensor(mag_cq_ola))

            # output is the original non-overlap-added slicq
            ys.append(tf.convert_to_tensor(mag_cq))

        return (
            tf.stack(
                [tf.expand_dims(x, axis=-1) for x in xs]
            ),
            tf.stack(
                [tf.expand_dims(y, axis=-1) for y in ys]
            ),
        )
