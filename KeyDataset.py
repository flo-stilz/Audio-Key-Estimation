# Key Dataset

import os
import torch
import numpy as np
import random
import time
import copy
from pathlib import Path
import math
import pathlib
import librosa
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from utils.key_signatures import KEY_SIGNATURE_MAP
from enum import Enum
import tensorflow as tf
import sox
import csv
import sys
import random
import time
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class FileExtension(Enum):
    WAV = 1
    MP3 = 2
    


class KeyDataset():

    def __init__(self, genre, opt):
        #self.hparams = hparams
        self.datasets = {}
        self.filenames = []
        self.genre = genre
        self.mel = {}
        self.key_labels = {}
        self.key_signature_id = {}
        self.genre_labels = {}
        self.tonic_labels = {}
        self.opt = opt
        
        
        
    def __len__(self):
        return len(self.filenames)
    
    def load_files(self, *dataset_loaders):
        # TODO: This method can be vectorized to run faster
        self.filenames = []
        for dataset_loader in dataset_loaders:
            print('Loading', dataset_loader.name, end='...', flush=True)
            if not isinstance(dataset_loader, DatasetLoader):
                print('skipped', flush=True)
                continue
            count = 0

            filenames = dataset_loader.get_filenames()
            for filename in filenames:
                #for i in range(2):
                pitch_shift = torch.tensor(0)
                # for finding songs that do not fit the format
                '''
                if dataset_loader.name == 'FSL10K':
                    if count%500==0:
                        print("Finished: "+str(count))
                    count+=1
                    keep = self.check_length(filename, pitch_shift)
                    if bytes.decode(filename.numpy()) in '/mnt/raid/fstilz/Key_estimation/Data/FSL10K/audio/wav/492982_5187472.wav.wav':
                        accept = False
                    else:
                        accept = self.check_key_confidence(dataset_loader.dataset_loc, filename)
                    
                    if keep and accept:
                        #self.filenames.append((filename, dataset_loader.name, pitch_shift))
                        pass
                    else:
                        with open('short_songs.txt', 'a') as f:
                            f.write("\n"+bytes.decode(filename.numpy()))
                 '''
                #else:
                #    self.filenames.append((filename, dataset_loader.name, pitch_shift))
                
                songs = []
                with open('short_songs.txt') as f:
                    songs = f.readlines()
                    for i in range(len(songs)):
                        songs[i] = songs[i].replace('\n', '')
                #print(songs)
                #print(len(songs))
                #sys.exit()
                keep = True
                for i in songs:
                    if bytes.decode(filename.numpy()) in i:
                        keep = False
                if keep:
                    self.filenames.append((filename, dataset_loader.name, pitch_shift))
                
                
                #self.filenames.append((filename, dataset_loader.name, pitch_shift))
            
    def decode_audio(self, file_path, file_extension: FileExtension):
        """
        Turns the passed binary audio into 'float32' tensors.

        The 'float32' tensors are normalized to the [-1.0, 1.0] range.
        Since all the data is single channel (mono), we drop the 'channels' axis from the array.

        Parameters
        ----------
        file_path : String tensor
            The file path to the song.
        file_extension : FileExtension
            The file extension of the song's file to read.

        Returns
        -------
        Tensor
        """
        file_path = bytes.decode(file_path.numpy()) # convert tf to string
        waveform, sample_rate = torchaudio.load(file_path)
        
        return waveform, sample_rate
    
    def check_length(self, file_path, pitch_shift):
        pitch_shift = int(tf.strings.to_number(pitch_shift).numpy())


        waveform, rate = self.decode_audio(file_path, FileExtension.WAV)

        if pitch_shift != 0:
            # create a transformer
            tfm = sox.Transformer()
            # shift the pitch up by 2 semitones
            tfm.pitch(pitch_shift)
            # transform an in-memory array and return an array
            waveform = tfm.build_array(input_array=waveform.numpy(), sample_rate_in=rate)
            waveform = tf.convert_to_tensor(waveform)

        w_length = waveform.shape[1]
        
        if w_length<100000:
            
            return False
        else:
            return True


    def load_dataset_handler(self, *dataset_loaders):
        for dataset_loader in dataset_loaders:
            if not isinstance(dataset_loader, DatasetLoader):
                continue
            self.datasets[dataset_loader.name] = dataset_loader

    def load_data_from_filename(self, filename, dataset_name, pitch_shift):
        key = dataset_name  # because keys cannot be vectors in this case

        return self.datasets[key].get_all(filename, pitch_shift, self.genre, self.opt)

    def import_data(self, *dataset_loaders):
        self.load_files(*dataset_loaders)
        self.load_dataset_handler(*dataset_loaders)
        
        # shuffling the dataset is done by shuffling the filenames (with their dataset name) and then importing them.
        # Reason is that shuffling the list of spectrograms is more computation/memory heavy.
        random.shuffle(self.filenames)
        #self.store_data_content()
        
        # check for invalid song files and remove if necessary
        '''
        remove = []
        for i in range(len(self.filenames)):
            if i%10==0:
                print("checked "+str(i)+" files")
            file_path = bytes.decode(self.filenames[i][0].numpy()) # convert tf to string
            try:
                waveform, sample_rate = torchaudio.load(file_path)
            except RuntimeError:
                remove.append(i)
                with open('short_songs.txt', 'a') as f:
                    f.write("\n"+bytes.decode(self.filenames[i][0].numpy()))
            #if bytes.decode(self.filenames[i].numpy())=="/mnt/raid/fstilz/Key_estimation/Data/KeyFinder/Luxury_Pool.mp3":
            #    remove.append(i)
        for i in range(len(self.filenames)):
            del self.filenames[i]
        '''
        self.store_content()
        #results = map(self.load_data_from_filename, map(lambda x: x[0], self.filenames), map(lambda x: x[1], self.filenames), map(lambda x: x[2], self.filenames))
        print("Length of Data: "+str(len(self.mel)))
        
    def load_content(self, idx):
    
        if int(idx)%50==0:
            print('loaded '+str(int(idx))+' files', flush=True)
        filename, dataset_name, pitch_shift = self.filenames[int(idx)]


        mel, key_labels, key_signature_id, genre = self.load_data_from_filename(filename, dataset_name, pitch_shift)

        print(key_labels)
        print(key_signature_id)
        print(self.filenames[int(idx)])        

        return mel, key_labels, key_signature_id, genre, int(idx)
       
        
    def store_content(self):
        index_list = torch.zeros(len(self.filenames))
        for i in range(len(index_list)):
            index_list[i] = i
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.save_files, index_list)
            for result in results:
                mel, key_labels, key_signature_id, genre, tonic_labels, idx = result
                self.mel[str(idx)] = mel
                self.key_labels[str(idx)] = key_labels
                self.key_signature_id[str(idx)] = key_signature_id
                self.genre_labels[str(idx)] = genre
                self.tonic_labels[str(idx)] = tonic_labels
                
        print('done', flush=True)
        
    def load_labels_from_filename(self, filename, dataset_name, mel):
        key = dataset_name  # because keys cannot be vectors in this case

        return self.datasets[key].get_all_labels(filename, self.genre, mel, self.opt)
        
    def save_files(self, idx):
        
        
        if int(idx)%10==0:
            print('loaded '+str(int(idx))+' files', flush=True)
        filename, dataset_name, pitch_shift = self.filenames[int(idx)]
        filename_str = bytes.decode(filename.numpy())
        
        melspectrogram = {}
        if self.opt.octaves==5 and self.opt.hop_length==1000:
            name = filename_str.replace(".wav","").replace(".mp3","")+"5oct_1k_hl.pt"
        elif self.opt.octaves==5:
            name = filename_str.replace(".wav","").replace(".mp3","")+"fmin64.pt"
        elif self.opt.octaves==7:
            name = filename_str.replace(".wav","").replace(".mp3","")+"7oct.pt"
        elif self.opt.octaves==8:
            name = filename_str.replace(".wav","").replace(".mp3","")+"8oct.pt"
        
        if self.opt.octaves==5:
            shape = 180
        elif self.opt.octaves==7:
            shape = 252
        elif self.opt.octaves==8:
            shape = 288
        else:
            shape = 360
            
        if os.path.exists(name):
            #print("saved!")
            melspectrogram = torch.load(name)
            #print(melspectrogram.shape)
            if melspectrogram.shape[1]==shape:# and melspectrogram.shape[2]==592:
                mel = melspectrogram
                key_labels, key_signature_id, genre, mel, tonic_labels = self.load_labels_from_filename(filename, dataset_name, mel=mel)
            else:
                mel, key_labels, key_signature_id, genre, tonic_labels = self.load_data_from_filename(filename, dataset_name, pitch_shift)
                torch.save(mel, name)
                
        else:
            mel, key_labels, key_signature_id, genre, tonic_labels = self.load_data_from_filename(filename, dataset_name, pitch_shift)
            
            torch.save(mel, name)
        
        # pad time axis if less than max value:
        if mel.shape[2]<592:
            diff = 592-mel.shape[2]
            mel = torch.cat((mel,torch.zeros([1,shape,diff])),dim=2)
    
        return mel, key_labels, key_signature_id, genre, tonic_labels, int(idx)
    
    
    def load_data_content(self, idx):
        ''' 
        if int(idx)%100==0:
            print('loaded '+str(int(idx))+' files', flush=True)
        '''
        filename, dataset_name, pitch_shift = self.filenames[int(idx)]
        filename_str = bytes.decode(filename.numpy())
        
        melspectrogram = {}
        name = filename_str.replace(".wav","").replace(".mp3","")+".pt"
        
        if os.path.exists(name):
            #print("saved!")
            melspectrogram = torch.load(name)
            if melspectrogram.shape[1]==360:
                mel = melspectrogram
                key_labels, key_signature_id, genre, mel = self.load_labels_from_filename(filename, dataset_name, mel=mel)
            else:
                mel, key_labels, key_signature_id, genre = self.load_data_from_filename(filename, dataset_name, pitch_shift)
                torch.save(mel, name)
        else:
            mel, key_labels, key_signature_id, genre = self.load_data_from_filename(filename, dataset_name, pitch_shift)
            
            torch.save(mel, name)

        return mel, key_labels, key_signature_id, genre, int(idx)
    
    
    def store_data_content(self):
        
        index_list = torch.zeros(len(self.filenames))
        for i in range(len(index_list)):
            index_list[i] = i
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.load_data_content, index_list)
            for result in results:
        
                mel, key_labels, key_signature_id, genre, idx = result
                
                self.mel[str(idx)] = mel
                self.key_labels[str(idx)] = key_labels
                self.key_signature_id[str(idx)] = key_signature_id
                self.genre_labels[str(idx)] = genre
        '''
        for i in range(len(self.filenames)):
            mel, key_labels, key_signature_id, genre, idx = self.load_data_content(i)
            self.mel[str(idx)] = mel
            self.key_labels[str(idx)] = key_labels
            self.key_signature_id[str(idx)] = key_signature_id
            self.genre_labels[str(idx)] = genre
        '''
            
            
    def __getitem__(self, idx):
        
        mel, key_labels, key_signature_id, genre, tonic_labels = self.mel[str(idx)], self.key_labels[str(idx)], self.key_signature_id[str(idx)], self.genre_labels[str(idx)], self.tonic_labels[str(idx)]

        #mel, key_labels, key_signature_id, genre, i = self.save_files(idx)
        
        #mel, key_labels, key_signature_id, genre = self.load_data_from_filename(filename, dataset_name, pitch_shift)
        #self.mel[str(idx)], self.key_labels[str(idx)], self.key_signature_id[str(idx)], self.genre_labels[str(idx)] = mel, key_labels, key_signature_id, genre
        if self.opt.local:
            seq_length = mel.shape[2]
            padded_seq_length = 28000
            padded_mel = torch.cat((mel,torch.zeros([mel.shape[0], mel.shape[1],padded_seq_length-mel.shape[2]])),dim=2)
            padded_key_labels = torch.cat((key_labels, torch.zeros([key_labels.shape[0], padded_seq_length-key_labels.shape[1]])),dim=1)
            padded_key_signature_id = torch.cat((key_signature_id, torch.zeros([key_signature_id.shape[0], padded_seq_length-key_signature_id.shape[1]])),dim=1)
            padded_genre = torch.cat((genre, torch.zeros([genre.shape[0], padded_seq_length-genre.shape[1]])),dim=1)
    
            item = {'mel': padded_mel}
            item['key_labels'] = padded_key_labels
            item['key_signature_id'] = padded_key_signature_id
            item['genre'] = padded_genre
            item['seq_length'] = seq_length
            
        else:
            item = {'mel': mel}
            item['key_labels'] = key_labels
            item['tonic_labels'] = tonic_labels
            item['key_signature_id'] = key_signature_id
            item['genre'] = genre
        
        return item

    
    
class DatasetLoader:
    """
    A class with the basic functionality of a dataset loader.
    Other dataset loaders have to inherit from this class and override methods that are not
    implemented yet or if their file system is different.

    Some functions from this class are taken from: https://www.tensorflow.org/tutorials/audio/simple_audio and modified.

    ...

    Attributes
    ----------
    size : int
        the number of songs (with keys) in this dataset

    Methods
    -------
    get_filenames():
        Returns all filenames of the audio data within this dataset.
    get_key_signature(file_path):
        Returns the key signature of the passed audio file.
    get_size():
        Returns the size of the dataset.
    decode_audio(audio_binary):
        Decodes the binary audio into 'float32' tensors.
    get_waveform_and_key_signature(file_path, pitch_shift=0):
        Returns the waveform and the key signature of the passed audio file.
    get_spectrogram(waveform):
        Turns the waveform into a spectrogram (via constant q-transform).
    get_spectrogram_and_label_id(audio, label):
        Returns the spectrogram and the label id (key id) of the passed waveform and label (key).
    preprocess_dataset():
        Returns a 'Pyorch Dataset' containing the spectrograms and keys of this dataset.
    """
    def __init__(self, dataset_loc):
        self.name = None
        self.dataset_loc = dataset_loc
        self.size = -1
        self.keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                     'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                     'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
                     'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']
        self.signature = []

    def get_filenames(self):
        raise NotImplementedError("The standard Dataset Loader has no allocated Dataset")

    def get_key_signature(self, file_path):
        raise NotImplementedError("The standard Dataset Loader has no allocated Dataset")

    def get_size(self):
        return self.size

    def decode_audio(self, file_path, file_extension: FileExtension):
        """
        Turns the passed binary audio into 'float32' tensors.

        The 'float32' tensors are normalized to the [-1.0, 1.0] range.
        Since all the data is single channel (mono), we drop the 'channels' axis from the array.

        Parameters
        ----------
        file_path : String tensor
            The file path to the song.
        file_extension : FileExtension
            The file extension of the song's file to read.

        Returns
        -------
        Tensor
        """
        file_path = bytes.decode(file_path.numpy()) # convert tf to string
        #try:
        waveform, sample_rate = torchaudio.load(file_path)
        '''
        except RuntimeError:
            with open('short_songs.txt', 'a') as f:
                f.write("\n"+file_path)
            waveform = np.zeros([1,500000])
            sample_rate = 22050
        #waveform, sample_rate = torchaudio.load(file_path)
        '''
        
        return waveform, sample_rate
    
    def get_all_labels(self, file_path, genre, mel, opt):
        
        time_length = mel.shape[2]-opt.window_size+1
        
        key_signature = self.get_key_signature(file_path)

        if genre:
            genre = self.get_genre(file_path)
        else:
            genre = tf.zeros([8])
        
        #print(len(key_signature[0]))
        
        if self.name=="Schubert Winterreise" and opt.local:
            for i in range(len(key_signature[0])):
                label_id = tf.argmax(key_signature[1][i] == self.keys) % 21
                key_labels_sub = KEY_SIGNATURE_MAP[label_id]
                
                key_signature_id_sub = tf.argmax(key_signature[1][i] == self.signature)  # 12 to get tonic instead of key signature
                key_signature_id_sub = tf.one_hot(key_signature_id_sub, 24)
                
                key_signature_id_sub = torch.tensor(key_signature_id_sub.numpy())
                key_labels_sub = torch.tensor(key_labels_sub.numpy())
                
                time_interval = bytes.decode(key_signature[0][i].numpy()).split("_")
                start = float(time_interval[0])
                end = float(time_interval[1])
                start_index = int(start*22.05)
                end_index = int(end*22.05)
                if i==0 and len(key_signature[0])>1:
                    start_cut = int(start*22.05)
                    repeats = int(int((end_index-start_index+(start*22.05)-int(start*22.05))*(opt.window_size/22.05))-(opt.window_size+1)+(opt.window_size/2)) # window_size/2 overlap from key change afterwards
                    previous_end_index = end_index
                if i==0:
                    start_cut = int(start*22.05)
                    repeats = int(int((end_index-start_index+(start*22.05)-int(start*22.05))*(opt.window_size/22.05))-(opt.window_size+1)) # window_size/2 overlap from key change afterwards
                elif i==(len(key_signature[0])-1):
                    end_cut = round(end*22.05)
                    repeats = int(int((end_index-start_index)*(opt.window_size/22.05))-(opt.window_size+1)+(opt.window_size/2-1)) # window_size/2 overlap from key change before
                elif i>0:
                    repeats = int(int((end_index-start_index)*(opt.window_size/22.05))-(opt.window_size+1)+(opt.window_size/2)+(opt.window_size/2-1)) # window_size/2 overlap from key change before and after
                    previous_end_index = end_index
            
                key_signature_id_sub = key_signature_id_sub.reshape(key_signature_id_sub.shape[0], 1).repeat(1,repeats)
                key_labels_sub = key_labels_sub.reshape(key_labels_sub.shape[0], 1).repeat(1,repeats)

                if i>0:
                    key_signature_id_s = torch.cat([key_signature_id_s, key_signature_id_sub], axis=1)
                    key_labels_s = torch.cat([key_labels_s, key_labels_sub],axis=1)
                if i==0:
                    key_signature_id_s = key_signature_id_sub
                    key_labels_s = key_labels_sub
                    
            # take beginning sequence without label out of the input
            # -> no sound contained in that part anyway -> would otherwise be a wrong label
            mel = mel[:, :, start_cut:]
            mel = mel[:,:,:key_labels_s.shape[1]+opt.window_size-1]
            key_signature_id = key_signature_id_s
            key_labels = key_labels_s
            assert key_labels.shape[1]==(mel.shape[2]-opt.window_size+1)
        
            
        else:
            label_id = tf.argmax(key_signature == self.keys) % 21
            key_labels = KEY_SIGNATURE_MAP[label_id]
    
            # TODO: key signature is wrong when a pitch shift is used!!
            key_signature_id = tf.argmax(key_signature == self.signature)  # 12 to get tonic instead of key signature
            key_signature_id = tf.one_hot(key_signature_id, 24)
            #print("Key_labels: "+str(key_labels))
            #print("Key_Signature_ID: "+str(key_signature_id))
            
            key_signature_id = torch.tensor(key_signature_id.numpy())
            key_labels = torch.tensor(key_labels.numpy())
        genre = torch.tensor(genre.numpy())
        
        #only needed for local key estimation
        if opt.local:
            if self.name!="Schubert Winterreise":
                key_signature_id = key_signature_id.reshape(key_signature_id.shape[0], 1).repeat(1,time_length)
                key_labels = key_labels.reshape(key_labels.shape[0], 1).repeat(1,time_length)
            genre = genre.reshape(genre.shape[0], 1).repeat(1,time_length)
            
        tonic_label = tf.math.argmax(key_signature == self.signature) % 12  # to get the tonic
        tonic_label = tf.one_hot(tf.cast(tonic_label, tf.int32), 12)   # one-hot encode the tonic
        tonic_label = torch.tensor(tonic_label.numpy())
        
            
        return key_labels, key_signature_id, genre, mel, tonic_label
    
        
    def get_all(self, file_path, pitch_shift, genre, opt):
        
        key_signature = self.get_key_signature(file_path)
        if genre:
            genre = self.get_genre(file_path)
        else:
            genre = tf.zeros([8])

        waveform, rate = self.decode_audio(file_path, FileExtension.WAV)

        w_length = waveform.shape[1]
        
        time_interval = 1 # means 1 sec
        song_length = math.ceil(w_length/rate) # song length in seconds
        hop_length = round(rate/22050 * opt.hop_length) # hop_length is 1000 for sample rate 22050 else adjusted accordingly to represent 1 song sec with 221 window size
        window_size = math.ceil(rate/hop_length)
        
        # C2 = 65Hz; C7 = 2093Hz; n_mels = 12(keys) * 5(octaves) * 2(bins per key) = 120
        #melspectrogram = librosa.feature.melspectrogram(y=waveform.numpy(), sr=44100, n_mels=120, fmin=63.57, fmax=2155.23,
        #                                                hop_length=w_length//592)
        melspectrogram = librosa.cqt(y=waveform.numpy(), sr=rate, hop_length=hop_length if opt.hop_length>0 else (w_length // opt.window_size + 1),# fmin=64,
                                     bins_per_octave=12 * 3, n_bins=12 * 3 * opt.octaves)
        # 44100Hz / 512(hop_size) => feature rate of ~86.1Hz
        #print(melspectrogram.shape)
        if melspectrogram.shape[0]==2:
            melspectrogram = melspectrogram[0]
        else:
            melspectrogram = melspectrogram.reshape(melspectrogram.shape[1], melspectrogram.shape[2])
        
        mel = torch.tensor(melspectrogram)
        mel = torch.abs(mel)
        mel = torch.log(1 + mel)  # log of the intensity
        
        if mel.shape[1] > opt.window_size:
            mel = mel[:, :opt.window_size]

        mel = mel.reshape(mel.shape[0], mel.shape[1], 1)
        
        if self.name=="Schubert Winterreise" and opt.local:

            for i in range(len(key_signature[0])):
                label_id = tf.argmax(key_signature[1][i] == self.keys) % 21
                key_labels_sub = KEY_SIGNATURE_MAP[label_id]
                
                key_signature_id_sub = tf.argmax(key_signature[1][i] == self.signature)  # 12 to get tonic instead of key signature
                key_signature_id_sub = tf.one_hot(key_signature_id_sub, 24)
                
                key_signature_id_sub = torch.tensor(key_signature_id_sub.numpy())
                key_labels_sub = torch.tensor(key_labels_sub.numpy())
                
                time_interval = bytes.decode(key_signature[0][i].numpy()).split("_")
                start = float(time_interval[0])
                end = float(time_interval[1])
                time_length_sub = round((end-start)*opt.window_size)
                
                key_signature_id_sub = key_signature_id_sub.reshape(key_signature_id_sub.shape[0], 1).repeat(1,time_length_sub)
                key_labels_sub = key_labels_sub.reshape(key_labels_sub.shape[0], 1).repeat(1,time_length_sub)
                if i>0:
                    key_signature_id_s = torch.cat([key_signature_id_s, key_signature_id_sub], axis=1)
                    key_labels_s = torch.cat([key_labels_s, key_labels_sub],axis=1)
                if i==0:
                    if start>=1.0:
                        # take beginning sequence without label out of the input
                        # -> no sound contained in that part anyway -> would otherwise be a wrong label
                        mel = mel[:, :, (int(start)*opt.window_size):]

                    key_signature_id_s = key_signature_id_sub
                    key_labels_s = key_labels_sub
                    
            key_signature_id = key_signature_id_s
            key_labels = key_labels_s
            
        else:

            label_id = tf.argmax(key_signature == self.keys) % 21
            key_labels = KEY_SIGNATURE_MAP[label_id]
    
            # TODO: key signature is wrong when a pitch shift is used!!
            key_signature_id = tf.argmax(key_signature == self.signature)  # 12 to get tonic instead of key signature
            key_signature_id = tf.one_hot(key_signature_id, 24)
            #print("Key_labels: "+str(key_labels))
            #print("Key_Signature_ID: "+str(key_signature_id))
            
            key_signature_id = torch.tensor(key_signature_id.numpy())
            key_labels = torch.tensor(key_labels.numpy())
        genre = torch.tensor(genre.numpy())
        
        #only needed for local key estimation
        time_length = mel.shape[1]-opt.window_size+1
        if opt.local:
            if self.name!="Schubert Winterreise":
                key_signature_id = key_signature_id.reshape(key_signature_id.shape[0], 1).repeat(1,time_length)
                key_labels = key_labels.reshape(key_labels.shape[0], 1).repeat(1,time_length)
            genre = genre.reshape(genre.shape[0], 1).repeat(1,time_length)
            
        tonic_label = tf.math.argmax(key_signature == self.signature) % 12  # to get the tonic
        tonic_label = tf.one_hot(tf.cast(tonic_label, tf.int32), 12)   # one-hot encode the tonic
        tonic_label = torch.tensor(tonic_label.numpy())
        
        return mel.reshape(1, mel.shape[0], mel.shape[1]).double(), key_labels, key_signature_id, genre, tonic_label
    


# ======================================================================================================================
class DatasetGiantStepsKeyLoader(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'GiantSteps Key'
        self.keys = ['Cb major', 'Gb major', 'Db major', 'Ab major', 'Eb major', 'Bb major', 'F major', 'C major',
                     'G major', 'D major', 'A major', 'E major', 'B major', 'F# major', 'C# major', '', '', '',
                     'D# major', 'G# major', 'A# major',
                     'Ab minor', 'Eb minor', 'Bb minor', 'F minor', 'C minor', 'G minor', 'D minor', 'A minor',
                     'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor', 'D# minor', 'A# minor',
                     'Cb minor', 'Db minor', 'Gb minor', '', '', '']
        self.signature = ['C minor', 'Db minor', 'D minor', 'Eb minor', 'E minor', 'F minor', 'Gb minor', 'G minor',
                          'Ab minor', 'A minor', 'Bb minor', 'B minor',
                          'C major', 'Db major', 'D major', 'Eb major', 'E major', 'F major', 'Gb major', 'G major',
                          'Ab major', 'A major', 'Bb major', 'B major']
        
        self.genres = ['breaks', 'techno', 'hip-hop', 'progressive-house', 'drum-and-bass', 'minimal', 'house', 'chill-out',
                       'deep-house', 'electro-house', 'trance', 'dubstep', 'tech-house', 'hard-dance', 'electronica', 'psy-trance',
                       'dj-tools', 'funk r&b', 'glitch-hop', 'hardcore hard-techno', 'indie-dance nu-disco', 'pop rock', 'reggae dub']
        self.a_genres = ['Classical', 'Rock', 'Pop', 'Folk', 'Metal', 'Electronic', 'Hip-Hop', 'R&B']
        self.genre_ids = [[],[],[21],[],[],[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20],[2], [16, 17, 22]]
        

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        filenames = tf.io.gfile.glob(str(data_dir) + '/audio/*.wav')
        #filenames = filenames[:8]
        filenames = tf.random.shuffle(filenames)
        self.size = len(filenames)
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        prior = tf.strings.reduce_join(parts[:-2], separator=os.path.sep)
        middle = tf.strings.join(["annotations", "key"], os.path.sep)
        name = tf.strings.regex_replace(parts[-1], "\\.wav", ".key")
        keypath = tf.strings.join([prior, middle, name], os.path.sep)
        return tf.strings.split(tf.io.read_file(keypath), '\t')[0]
    
    def get_genre(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        prior = tf.strings.reduce_join(parts[:-2], separator=os.path.sep)
        middle = tf.strings.join(["annotations", "genre"], os.path.sep)
        name = tf.strings.regex_replace(parts[-1], "\\.wav", ".genre")
        genrepath = tf.strings.join([prior, middle, name], os.path.sep)
        subgenre = tf.strings.split(tf.strings.split(tf.io.read_file(genrepath), '\t')[0], '\n')[0]
        subgenre_idx = tf.math.argmax(subgenre == self.genres)
        subgenre_idx = tf.dtypes.cast(subgenre_idx, tf.int32)
        for a_idx in range(len(self.genre_ids)):
            for idx in self.genre_ids[a_idx]:
                if idx == subgenre_idx:
                    genre = tf.one_hot(a_idx, len(self.a_genres))
                    break
        
        return genre


# ======================================================================================================================
class DatasetGiantStepsMTGKeyLoader(DatasetGiantStepsKeyLoader):
    def __init__(self, dataset_loc, data_type="train"):
        super().__init__(dataset_loc)
        self.name = 'GiantSteps MTG Key'
        self.keys = ['cb major', 'gb major', 'db major', 'ab major', 'eb major', 'bb major', 'f major', 'c major',
                     'g major', 'd major', 'a major', 'e major', 'b major', 'f# major', 'c# major', '', '', '',
                     'd# major', 'g# major', 'a# major',
                     'ab minor', 'eb minor', 'bb minor', 'f minor', 'c minor', 'g minor', 'd minor', 'a minor',
                     'e minor', 'b minor', 'f# minor', 'c# minor', 'g# minor', 'd# minor', 'a# minor',
                     'cb minor', 'db minor', 'gb minor', '', '', '']
        self.signature = ['c minor', 'c# minor', 'd minor', 'd# minor', 'e minor', 'f minor', 'f# minor', 'g minor',
                          'g# minor', 'a minor', 'a# minor', 'b minor',
                          'c major', 'c# major', 'd major', 'd# major', 'e major', 'f major', 'f# major', 'g major',
                          'g# major', 'a major', 'a# major', 'b major']

        self.genres = ['breaks', 'techno', 'hip-hop', 'progressive house', 'drum & bass', 'minimal', 'house', 'chill out',
                       'deep house', 'electro house', 'trance', 'dubstep', 'tech house', 'hard dance', 'electronica', 'psy-trance'
                       '', '', '', '', '', '', '']
        self.a_genres = ['Classical', 'Rock', 'Pop', 'Folk', 'Metal', 'Electronic', 'Hip-Hop', 'R&B']
        self.genre_ids = [[],[],[],[],[],[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],[2],[]]
        self.type = data_type
        
    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        
        filenames = tf.io.gfile.glob(str(data_dir) + '/audio/*.wav')
        #filenames = filenames[:8]
        filenames = tf.random.shuffle(filenames)

        valids = tf.map_fn(fn=self.get_key_signature, elems=filenames)
        valids_mask = tf.map_fn(fn=lambda key: not tf.strings.regex_full_match(key, ".*/.*"), elems=valids,
                                fn_output_signature=tf.bool)
        filenames = tf.boolean_mask(filenames, valids_mask)
        
        if self.type == "train":
            filenames = filenames[:round(len(filenames)*0.7)]
        elif self.type == "val":
            filenames = filenames[round(len(filenames)*0.7):]
            
        self.size = len(filenames)
        return filenames


# ======================================================================================================================
class DatasetTheBeatlesLoader(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'The Beatles'
        # This does not differentiate between modes and only gives the tonic

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.mp3')
        filenames = tf.random.shuffle(filenames)
        self.size = len(filenames)
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        prior = tf.strings.reduce_join(parts[:-2], separator=os.path.sep)
        middle = tf.strings.join(["The_Beatles_Annotations", "keylab", "The_Beatles", parts[-2]], os.path.sep)
        name = tf.strings.regex_replace(parts[-1], "\\.mp3", ".lab")
        keypath = tf.strings.join([prior, middle, name], os.path.sep)
        key_content_lines = tf.strings.split(tf.io.read_file(keypath), '\n')
        key_content_lines = tf.boolean_mask(key_content_lines,
                                            tf.strings.regex_full_match(key_content_lines, ".*\tKey\t.*"))
        key_content_lines = tf.strings.split(key_content_lines, '\t')

        return key_content_lines  # tf.strings.split(keyfile, '\t')[0]

    def get_waveform_and_key_signature(self, file_path, pitch_shift=0):
        audio_binary = tf.io.read_file(file_path)
        print(audio_binary.shape)
        waveform = self.decode_audio(audio_binary, FileExtension.MP3)

        key_signature_parts = self.get_key_signature(file_path)
        key_part = np.random.randint(int(key_signature_parts.shape[0] or 1))
        audio_from = int(float(key_signature_parts[key_part][0])) * 44100
        audio_to = int(float(key_signature_parts[key_part][1])) * 44100
        key_signature = key_signature_parts[key_part][3]

        audio_length = 800000
        duration = audio_to - audio_from - audio_length if audio_to - audio_from - audio_length > 0 else 1
        audio_start = audio_from + tf.random.uniform(shape=[], maxval=duration, dtype=tf.int32)
        audio_end = audio_start + audio_length if audio_start + audio_length < audio_to else audio_to

        waveform = waveform[audio_start:audio_end]
        zero_padding = tf.zeros(
            [audio_length] - tf.shape(waveform),
            dtype=tf.float32)
        # Cast the waveform tensors' dtype to float32
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Concatenate the waveform with 'zero_padding', which ensures all audio
        # clips are of the same length.
        equal_length = tf.concat([waveform, zero_padding], 0)

        return equal_length, tf.strings.split(key_signature, ':')[0]


# ======================================================================================================================
class DatasetSchubertWinterreiseLoader(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'Schubert Winterreise'
        self.keys = ['Cb:maj', 'Gb:maj', 'Db:maj', 'Ab:maj', 'Eb:maj', 'Bb:maj', 'F:maj', 'C:maj',
                     'G:maj', 'D:maj', 'A:maj', 'E:maj', 'B:maj', 'F#:maj', 'C#:maj', '', '', '',
                     'D#:maj', 'G#:maj', 'A#:maj',
                     'Ab:min', 'Eb:min', 'Bb:min', 'F:min', 'C:min', 'G:min', 'D:min', 'A:min',
                     'E:min', 'B:min', 'F#:min', 'C#:min', 'G#:min', 'D#:min', 'A#:min',
                     'Cb:min', 'Db:min', 'Gb:min', '', '', '']
        self.signature = ['C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min',
                          'A:min', 'A#:min', 'B:min',
                          'C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj',
                          'A:maj', 'A#:maj', 'B:maj',
                          'C:min', 'Db:min', 'D:min', 'Eb:min', 'E:min', 'F:min', 'Gb:min', 'G:min', 'Ab:min',
                          'A:min', 'Bb:min', 'B:min',
                          'C:maj', 'Db:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'Gb:maj', 'G:maj', 'Ab:maj',
                          'A:maj', 'Bb:maj', 'B:maj']
        
        self.a_genres = ['Classical', 'Rock', 'Pop', 'Folk', 'Metal', 'Electronic', 'Hip-Hop', 'R&B']

        self.song_list, self.global_keys = self.get_global_keys()
        self.local_song_list, self.local_keys = self.get_local_keys()

    def get_global_keys(self):
        prior = self.dataset_loc
        middle = "02_Annotations"
        name = "ann_audio_globalkey.csv"
        keypath = tf.strings.join([prior, middle, name], os.path.sep)
        lines = tf.strings.split(tf.io.read_file(keypath), '\n')[1:]
        lines = tf.strings.split(lines, ';')
        lines = tf.strings.regex_replace(lines, "\"", "")
        lines = tf.strings.regex_replace(lines, "\r", "")

        songs = tf.strings.reduce_join(lines[:, :2], axis=-1, separator='_')
        keys = tf.squeeze(lines[:, -1:], axis=1)
        return songs, keys
    
    def get_local_keys(self):
        prior = self.dataset_loc
        middle = "02_Annotations"
        last = "ann_audio_localkey-ann3"
        folder = tf.strings.join([prior, middle, last], os.path.sep)
        songs = []
        local_keys_list = []
        for root, subdirectories, files in os.walk(bytes.decode(folder.numpy())):
            for file in files:
                songs.append(file.split("/")[-1].replace(".csv",""))
                lines = tf.strings.split(tf.io.read_file(os.path.join(bytes.decode(folder.numpy()),file)), '\n')[1:]
                lines = tf.strings.split(lines, ';')
                lines = tf.strings.regex_replace(lines, "\"", "")
                lines = tf.strings.regex_replace(lines, "\r", "")
                time_interval = tf.strings.reduce_join(lines[:, :2], axis=-1, separator='_')
                keys = tf.squeeze(lines[:, -1:], axis=1)
                #print(time_interval)
                #print(keys)
                local_keys = (time_interval, keys)
                local_keys_list.append(local_keys)
                
                
        return tf.convert_to_tensor(songs), local_keys_list
         
        

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        filenames = tf.io.gfile.glob(str(data_dir) + '/01_RawData/audio_wav/*.wav')
        filenames = tf.random.shuffle(filenames)
        self.size = len(filenames)
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        song_name = tf.strings.regex_replace(parts[-1], "\\.wav", "")
        #return self.local_keys[tf.math.argmax(self.local_song_list == song_name)]
        return self.global_keys[tf.math.argmax(self.song_list == song_name)]
    
    def get_genre(self, file_path):
        # We know it is Classical:
        genre = tf.one_hot(0, len(self.a_genres))
                    
        
        return genre


# ======================================================================================================================
class DatasetGTZANLoader(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'GTZAN'
        self.keys = ['', '', '', '', '', '', '8', '3', '10', '5', '0', '7', '2', '9', '4', '', '', '', '6', '11', '1',
                     '', '', '', '20', '15', '22', '17', '12', '19', '14', '21', '16', '23', '18', '13', '', '', '']
        self.signature = ['15', '16', '17', '18', '19', '20', '21', '22', '23', '12', '13', '14',
                          '3',   '4',  '5',  '6',  '7',  '8',  '9', '10', '11',  '0',  '1',  '2']

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        filenames = tf.io.gfile.glob(str(data_dir) + '/genres_original/*/*.wav')
        filenames = tf.random.shuffle(filenames)

        valids = tf.map_fn(fn=self.get_key_signature, elems=filenames)
        valids_mask = tf.map_fn(fn=lambda key: key != "-1", elems=valids,
                                fn_output_signature=tf.bool)
        filenames = tf.boolean_mask(filenames, valids_mask)

        self.size = len(filenames)
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        prior = self.dataset_loc
        middle = tf.strings.join(["gtzan_key", "genres", parts[-2]], os.path.sep)
        name = tf.strings.regex_replace(parts[-1], "\\.wav", ".lerch.txt")
        keypath = tf.strings.join([prior, middle, name], os.path.sep)
        return tf.io.read_file(keypath)


# ======================================================================================================================
class DatasetLoaderYouTubeScraped(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)

        # TODO: maybe use 'os.path.sep' instead of '/', and maybe 'dataset_loc' already ends with os.path.sep?
        with open(dataset_loc + '/__youtube_similarities.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
        self.data = tf.convert_to_tensor(data)
        self.threshold = 0.6

    def decode_audio(self, audio_binary, file_extension: FileExtension):
        return super().decode_audio(audio_binary, FileExtension.MP3)

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        filenames = tf.io.gfile.glob(str(data_dir) + '/*.mp3')  # TODO: maybe use 'os.path.sep' instead of '/'
        filenames = tf.random.shuffle(filenames)

        song_names = tf.strings.split(
            input=filenames,
            sep=os.path.sep)[:, -1:]
        too_long = ['Daft Punk Solar Sailer', 'The Chemical Brothers Dig Your Own Hole', 'Phaeleh Fallen Light']
        valids = tf.map_fn(fn=lambda filename: tf.strings.regex_replace(filename, "\\.mp3", ""), elems=song_names)
        valids_mask = tf.map_fn(fn=lambda song_name: self.get_score(song_name) >= self.threshold and
                                song_name not in too_long, elems=valids,
                                fn_output_signature=tf.bool)
        filenames = tf.boolean_mask(filenames, valids_mask)

        self.size = len(filenames)
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        name = tf.strings.regex_replace(parts[-1], "\\.mp3", "")
        mask = self.data[:, 0] == name
        song_data = tf.boolean_mask(self.data, mask)
        return song_data[0, 2]  # the key of the song

    def get_score(self, song_name: str):
        mask = self.data[:, 0] == tf.constant(song_name)  # song_name has to be casted from utf-8 to no encoding
        song_data = tf.boolean_mask(self.data, mask)
        score = tf.strings.to_number(song_data[0, 1])
        return score  # the similarity score of the song


# ======================================================================================================================
class DatasetKeyFinderLoader(DatasetLoaderYouTubeScraped):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'KeyFinder'
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Abm', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm', 'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m',
                     'Cbm', 'Dbm', 'Gbm', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
                          'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']


# ======================================================================================================================
class DatasetMcGillBillboardLoader(DatasetLoaderYouTubeScraped):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'McGill Billboard'
        # This does not differentiate between modes and only gives the tonic
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Abm', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm', 'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m',
                     'Cbm', 'Dbm', 'Gbm', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
                          'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']


# ======================================================================================================================
class DatasetTonalityClassicalDBLoader(DatasetLoaderYouTubeScraped):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'Tonality Classical DB'
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Abm', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm', 'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m',
                     'Cbm', 'Dbm', 'Gbm', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
                          'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']


# ======================================================================================================================
class DatasetGuitarSetLoader(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'GuitarSet'
        self.keys = ['Cb:major', 'Gb:major', 'Db:major', 'Ab:major', 'Eb:major', 'Bb:major', 'F:major', 'C:major',
                     'G:major', 'D:major', 'A:major', 'E:major', 'B:major', 'F#:major', 'C#:major', '', '', '',
                     'D#:major', 'G#:major', 'A#:major',
                     'Ab:minor', 'Eb:minor', 'Bb:minor', 'F:minor', 'C:minor', 'G:minor', 'D:minor', 'A:minor',
                     'E:minor', 'B:minor', 'F#:minor', 'C#:minor', 'G#:minor', 'D#:minor', 'A#:minor',
                     'Cb:minor', 'Db:minor', 'Gb:minor', '', '', '']
        self.signature = ['C:minor', 'C#:minor', 'D:minor', 'D#:minor', 'E:minor', 'F:minor', 'F#:minor', 'G:minor',
                          'G#:minor', 'A:minor', 'A#:minor', 'B:minor',
                          'C:major', 'C#:major', 'D:major', 'D#:major', 'E:major', 'F:major', 'F#:major', 'G:major',
                          'G#:major', 'A:major', 'A#:major', 'B:major',
                          'C:minor', 'Db:minor', 'D:minor', 'Eb:minor', 'E:minor', 'F:minor', 'Gb:minor', 'G:minor',
                          'Ab:minor', 'A:minor', 'Bb:minor', 'B:minor',
                          'C:major', 'Db:major', 'D:major', 'Eb:major', 'E:major', 'F:major', 'Gb:major', 'G:major',
                          'Ab:major', 'A:major', 'Bb:major', 'B:major']

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        filenames = tf.io.gfile.glob(str(data_dir) + '/audio_mono-mic/*.wav')
        filenames = tf.random.shuffle(filenames)
        self.size = len(filenames)
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        prior = self.dataset_loc
        middle = "annotations"
        name = tf.strings.regex_replace(parts[-1], "_mic\\.wav", ".jams")
        keypath = tf.strings.join([prior, middle, name], os.path.sep)
        f = open(keypath.numpy())
        data = json.load(f)
        return tf.convert_to_tensor(data['annotations'][-1]['data'][0]['value'])
    
# ======================================================================================================================
class DatasetFSL10KLoader(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'FSL10K'
        
        self.keys = ['Cb major', 'Gb major', 'Db major', 'Ab major', 'Eb major', 'Bb major', 'F major', 'C major',
                     'G major', 'D major', 'A major', 'E major', 'B major', 'F# major', 'C# major', '', '', '',
                     'D# major', 'G# major', 'A# major',
                     'Ab minor', 'Eb minor', 'Bb minor', 'F minor', 'C minor', 'G minor', 'D minor', 'A minor',
                     'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor', 'D# minor', 'A# minor',
                     'Cb minor', 'Db minor', 'Gb minor', '', '', '']
        self.signature = ['C minor', 'C# minor', 'D minor', 'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor',
                          'G# minor', 'A minor', 'A# minor', 'B minor',
                          'C major', 'C# major', 'D major', 'D# major', 'E major', 'F major', 'F# major', 'G major',
                          'G# major', 'A major', 'A# major', 'B major']

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        filenames = tf.io.gfile.glob(str(data_dir) + '/audio/wav/*.wav')
        filenames = tf.random.shuffle(filenames)
        self.size = len(filenames)
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        prior = self.dataset_loc
        middle = "ac_analysis"
        
        if 'aiff' in bytes.decode(parts[-1].numpy()):
            name = tf.strings.regex_replace(parts[-1], ".aiff.wav", "_analysis.json")

        else:
            name = tf.strings.regex_replace(parts[-1], ".wav.wav", "_analysis.json")
        keypath = tf.strings.join([prior, middle, name], os.path.sep)
        f = open(keypath.numpy())
        data = json.load(f)
        return tf.convert_to_tensor(data['tonality'])
    
# ======================================================================================================================
class DatasetBeatlesLoader(DatasetLoaderYouTubeScraped):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'The Beatles Dataset'
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Ab:minor', 'Eb:mino', 'Bb:minor', 'F:minor', 'C:minor', 'G:minor', 'D:minor', 'A:minor', 'E:minor', 'B:minor', 'F#:minor', 'C#:minor', 'G#:minor', 'D#:minor', 'A#:minor',
                     'Cb:minor', 'Db:minor', 'Gb:minor', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'C:minor', 'C#:minor', 'D:minor', 'D#:minor', 'E:minor', 'F:minor', 'F#:minor', 'G:minor', 'G#:minor', 'A:minor', 'A#:minor', 'B:minor',
                          'C:minor', 'Db:minor', 'D:minor', 'Eb:minor', 'E:minor', 'F:minor', 'Gb:minor', 'G:minor', 'Ab:minor', 'A:minor', 'Bb:minor', 'B:minor']
        
# ======================================================================================================================
class DatasetKingCaroleLoader(DatasetLoaderYouTubeScraped):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'King Carole Dataset'
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Ab:minor', 'Eb:mino', 'Bb:minor', 'F:minor', 'C:minor', 'G:minor', 'D:minor', 'A:minor', 'E:minor', 'B:minor', 'F#:minor', 'C#:minor', 'G#:minor', 'D#:minor', 'A#:minor',
                     'Cb:minor', 'Db:minor', 'Gb:minor', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'C:minor', 'C#:minor', 'D:minor', 'D#:minor', 'E:minor', 'F:minor', 'F#:minor', 'G:minor', 'G#:minor', 'A:minor', 'A#:minor', 'B:minor',
                          'C:minor', 'Db:minor', 'D:minor', 'Eb:minor', 'E:minor', 'F:minor', 'Gb:minor', 'G:minor', 'Ab:minor', 'A:minor', 'Bb:minor', 'B:minor']

# ======================================================================================================================
class DatasetQueenLoader(DatasetLoaderYouTubeScraped):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'Queen Dataset'
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Ab:minor', 'Eb:mino', 'Bb:minor', 'F:minor', 'C:minor', 'G:minor', 'D:minor', 'A:minor', 'E:minor', 'B:minor', 'F#:minor', 'C#:minor', 'G#:minor', 'D#:minor', 'A#:minor',
                     'Cb:minor', 'Db:minor', 'Gb:minor', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'C:minor', 'C#:minor', 'D:minor', 'D#:minor', 'E:minor', 'F:minor', 'F#:minor', 'G:minor', 'G#:minor', 'A:minor', 'A#:minor', 'B:minor',
                          'C:minor', 'Db:minor', 'D:minor', 'Eb:minor', 'E:minor', 'F:minor', 'Gb:minor', 'G:minor', 'Ab:minor', 'A:minor', 'Bb:minor', 'B:minor']

# ======================================================================================================================
class DatasetZweieckLoader(DatasetLoaderYouTubeScraped):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        self.name = 'Zweieck Dataset'
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Ab:minor', 'Eb:mino', 'Bb:minor', 'F:minor', 'C:minor', 'G:minor', 'D:minor', 'A:minor', 'E:minor', 'B:minor', 'F#:minor', 'C#:minor', 'G#:minor', 'D#:minor', 'A#:minor',
                     'Cb:minor', 'Db:minor', 'Gb:minor', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'C:minor', 'C#:minor', 'D:minor', 'D#:minor', 'E:minor', 'F:minor', 'F#:minor', 'G:minor', 'G#:minor', 'A:minor', 'A#:minor', 'B:minor',
                          'C:minor', 'Db:minor', 'D:minor', 'Eb:minor', 'E:minor', 'F:minor', 'Gb:minor', 'G:minor', 'Ab:minor', 'A:minor', 'Bb:minor', 'B:minor']

# ======================================================================================================================
class DatasetPopularSongsLoader(DatasetLoader):
    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)
        data_dir = pathlib.Path(self.dataset_loc)
        self.name = 'Popular Songs Dataset'
        self.keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     '', '', '', 'D#', 'G#', 'A#',
                     'Abm', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm', 'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m',
                     'Cbm', 'Dbm', 'Gbm', '', '', '']
        self.signature = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                          'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
                          'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
                          'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']
        
        self.a_genres = ['Classical', 'Rock', 'Pop', 'Folk', 'Metal', 'Electronic', 'Hip-Hop', 'R&B']

        # TODO: maybe use 'os.path.sep' instead of '/', and maybe 'dataset_loc' already ends with os.path.sep?
        folders = ["SubA", "SubA#m", "SubAb", "SubAbm", "SubAm", "SubB", "SubBb", "SubBbm", "SubBm", "SubC", "SubC#", "SubC#m", "SubCb", "SubCm", "SubD", "SubD#m", "SubDb", "SubDm", "SubE", "SubEb", "SubEbm", "SubEm", "SubF", "SubF#", "SubF#m", "SubFm", "SubG", "SubG#m", "SubGb", "SubGm"]
        genre = ["Rock", "Pop", "Classical", "Metal", "Folk", "RandB", "Hip-Hop"]
        #genre = ["Rock"]
        data = []
        for j in genre:
            if j == "Rock" or j== "Pop":
                for i in folders:
                    if os.path.exists(dataset_loc+'/'+j+'/'+i+'/__youtube_similarities.csv'):
                        with open(dataset_loc+'/'+j+'/'+i+'/__youtube_similarities.csv', newline='', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            data = data + list(reader)
            elif j == "Classical":
                with open(dataset_loc+'/'+j+'/__youtube_similarities.csv', newline='', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            data = data + list(reader)
            else:
                for i in range(1,4):
                    if os.path.exists(dataset_loc+'/'+j+'/'+j+str(i)+'/__youtube_similarities.csv'):
                        with open(dataset_loc+'/'+j+'/'+j+str(i)+'/__youtube_similarities.csv', newline='', encoding='utf-8') as f:
                            reader = csv.reader(f)
                            data = data + list(reader)
        self.data = tf.convert_to_tensor(data)
        self.threshold = 0.8 # might want to increase further to ensure correctness
        
    def decode_audio(self, audio_binary, file_extension: FileExtension):
        return super().decode_audio(audio_binary, FileExtension.MP3)

    def get_filenames(self):
        data_dir = pathlib.Path(self.dataset_loc)
        folders = ["SubA", "SubA#m", "SubAb", "SubAbm", "SubAm", "SubB", "SubBb", "SubBbm", "SubBm", "SubC", "SubC#", "SubC#m", "SubCb", "SubCm", "SubD", "SubD#m", "SubDb", "SubDm", "SubE", "SubEb", "SubEbm", "SubEm", "SubF", "SubF#", "SubF#m", "SubFm", "SubG", "SubG#m", "SubGb", "SubGm"]
        genre = ["Rock", "Pop", "Classical", "Metal", "Folk", "RandB", "Hip-Hop"]
        #genre = ["Rock"]
        for j in genre:
            if j == "Rock" or j == "Pop":
                for i in range(len(folders)):
                    data_dir_sub = os.path.join(data_dir,j+"/"+folders[i])
                    if i==0 and j=="Rock":
                        filenames = tf.io.gfile.glob(str(data_dir_sub) + '/*.mp3')
                    else:
                        filenames_sub = tf.io.gfile.glob(str(data_dir_sub) + '/*.mp3')
                        filenames = tf.concat([filenames, filenames_sub],axis=0)
            elif j == "Classical":
                data_dir_sub = os.path.join(data_dir, j)
                filenames_sub = tf.io.gfile.glob(str(data_dir_sub) + '/*.mp3')
                filenames = tf.concat([filenames, filenames_sub], axis=0)
            else:
                for i in range(1,4):
                    if os.path.exists(os.path.join(data_dir,j+'/'+j+str(i))):
                        data_dir_sub = os.path.join(data_dir,j+'/'+j+str(i))
                        filenames_sub = tf.io.gfile.glob(str(data_dir_sub)+ '/*.mp3')
                        filenames = tf.concat([filenames, filenames_sub], axis=0)
        
        filenames2 = []
        print("\nTotal amount of files in Popular Songs: "+str(len(filenames)))
        for file in filenames:
            if os.path.getsize(bytes.decode(file.numpy()))<5000000: # limit to 5MB files
                filenames2.append(file)
        filenames = tf.convert_to_tensor(filenames2)
        print("Files with optimal mem: "+str(len(filenames)))
        #filenames = tf.random.shuffle(filenames)

        song_names = tf.strings.split(
            input=filenames,
            sep=os.path.sep)[:, -1:]
        too_long = ['Daft Punk Solar Sailer', 'The Chemical Brothers Dig Your Own Hole', 'Phaeleh Fallen Light']
        valids = tf.map_fn(fn=lambda filename: tf.strings.regex_replace(filename, "\\.mp3", ""), elems=song_names)
        valids_mask = tf.map_fn(fn=lambda song_name: self.get_score(song_name) >= self.threshold and
                                song_name not in too_long, elems=valids,
                                fn_output_signature=tf.bool)
        filenames = tf.boolean_mask(filenames, valids_mask)
        print("Files with appropriate score: "+str(len(filenames)))
        filenames = tf.random.shuffle(filenames)
        self.size = len(filenames)
        print("Actual Dataset size: "+ str(self.size))
        return filenames

    def get_key_signature(self, file_path):
        parts = tf.strings.split(
            input=file_path,
            sep=os.path.sep)
        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        name = tf.strings.regex_replace(parts[-1], "\\.mp3", "")
        mask = self.data[:, 0] == name
        song_data = tf.boolean_mask(self.data, mask)
        return song_data[0, 2]  # the key of the song

    def get_score(self, song_name: str):
        mask = self.data[:, 0] == tf.constant(song_name)  # song_name has to be casted from utf-8 to no encoding
        song_data = tf.boolean_mask(self.data, mask)
        score = tf.strings.to_number(song_data[0, 1])
        return score  # the similarity score of the song

        
    def get_genre(self, file_path):
        
        file_path = bytes.decode(file_path.numpy())
        for i in range(len(self.a_genres)):
            if self.a_genres[i] in file_path:
                genre = tf.one_hot(i, len(self.a_genres))
                break
    
        return genre