import pathlib
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import re
import json


def get_songs_from_keyfinder(songs_filename: str):
    #file = tf.io.read_file(songs_filename)
    data = pd.read_csv(songs_filename+'/KeyFinderV2Dataset.csv')
    titles = data['TITLE']
    data = np.array(data)
    data = data[~pd.isnull(titles)]
    #lines = tf.strings.split(input=song_list, sep='\r\n')
    songs = tf.convert_to_tensor(data[:,1]) # titles
    keys = tf.convert_to_tensor(data[:,2]) # keys
    #parts = tf.strings.split(input=lines, sep=' ')
    #songs = tf.strings.reduce_join(parts[:, :-1], axis=-1, separator=' ')
    #keys = tf.squeeze(parts[:, -1:], axis=-1)
    
    return songs, keys


def get_title_from_billboard(song_filename: str):
    data = tf.io.read_file(song_filename)
    lines = tf.strings.split(input=data, sep='\n')
    title = tf.strings.split(input=lines[0], sep=':')[1]
    artist = tf.strings.split(input=lines[1], sep=':')[1]
    tonic = tf.strings.split(input=lines[3], sep=':')[1]
    title = tf.strings.strip(title)
    artist = tf.strings.strip(artist)
    tonic = tf.strings.strip(tonic)
    song_query = artist + ' ' + title
    return tf.convert_to_tensor([song_query, tonic])


def get_songs_from_billboard(folder_name: str):
    data_dir = pathlib.Path(folder_name)
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.txt')
    filenames = tf.convert_to_tensor(sorted(filenames))

    songs_data = tf.map_fn(fn=get_title_from_billboard, elems=filenames,
                           fn_output_signature=tf.TensorSpec(shape=[2,], dtype=tf.string))
    song_queries = songs_data[:, 0]
    song_keys = songs_data[:, 1]

    return song_queries, song_keys


"""def get_name(filename):
    parts = tf.strings.split(input=filename, sep=os.path.sep)
    name = tf.strings.split(input=parts[-1], sep='-')[1]
    name = tf.strings.regex_replace(name, "GTKeys.txt", "")
    return 'Robbie Williams ' + name


def get_songs_from_robbie_williams(folder_name: str):
    data_dir = pathlib.Path(folder_name)
    filenames = tf.io.gfile.glob(str(data_dir) + '/keys/*/*.txt')
    filenames = tf.convert_to_tensor(sorted(filenames))

    song_queries = tf.map_fn(fn=get_name, elems=filenames)

    return song_queries, None"""


def get_title_and_key_from_tonality(song_filename: str):
    filename = tf.strings.split(song_filename, os.path.sep)[-1]
    parts = tf.strings.split(filename, ' - ')
    title = tf.strings.regex_replace(parts[0], '_', ' ')
    key = tf.strings.regex_replace(parts[1], '.txt', '')
    return tf.convert_to_tensor([title, key])


def get_songs_from_tonality(folder_name: str):
    data_dir = pathlib.Path(folder_name)
    filenames = tf.io.gfile.glob(str(data_dir) + '/keys/*.txt')
    filenames = tf.convert_to_tensor(sorted(filenames))

    songs_data = tf.map_fn(fn=get_title_and_key_from_tonality, elems=filenames,
                           fn_output_signature=tf.TensorSpec(shape=[2, ], dtype=tf.string))
    song_queries = songs_data[:, 0]
    song_keys = songs_data[:, 1]

    return song_queries, song_keys

def get_songs_from_beatles(folder_name: str):
    #data_dir = pathlib.Path(folder_name)
    #folder = tf.io.gfile.glob(str(data_dir) + '/keylab/The_Beatles')
    folder = os.path.join(folder_name, 'keylab/The_Beatles')
    song_queries = []
    song_keys = []
    for root, subdirectories, files in os.walk(folder):
        for subdirectory in subdirectories:
            for root, subdirectory, files in os.walk(os.path.join(folder,subdirectory)):
                for file in files:
                    if ".lab" in file:
                        song_queries.append(file[5:].replace(".lab","").replace("_"," "))
                        loc = os.path.join(root, file)
                        key = open(loc, "r")
                        key = key.read()
                        song_keys.append(key.split("Key\t")[1].split("\n")[0])
     
        
    return tf.convert_to_tensor(song_queries), tf.convert_to_tensor(song_keys)

def get_songs_from_king_carole(folder_name: str):
    #data_dir = pathlib.Path(folder_name)
    #folder = tf.io.gfile.glob(str(data_dir) + '/keylab/The_Beatles')
    folder = os.path.join(folder_name, 'keylab/Carole_King')
    song_queries = []
    song_keys = []
    for root, subdirectories, files in os.walk(folder):
        for subdirectory in subdirectories:
            for root, subdirectory, files in os.walk(os.path.join(folder,subdirectory)):
                for file in files:
                    if ".lab" in file:
                        song_queries.append(file[5:].replace(".lab","").replace("_"," "))
                        loc = os.path.join(root, file)
                        key = open(loc, "r")
                        key = key.read()
                        song_keys.append(key.split("Key\t")[1].split("\n")[0])
     
        
    return tf.convert_to_tensor(song_queries), tf.convert_to_tensor(song_keys)

def get_songs_from_queen(folder_name: str):
    #data_dir = pathlib.Path(folder_name)
    #folder = tf.io.gfile.glob(str(data_dir) + '/keylab/The_Beatles')
    folder = os.path.join(folder_name, 'keylab/Queen')
    song_queries = []
    song_keys = []
    for root, subdirectories, files in os.walk(folder):
        for subdirectory in subdirectories:
            for root, subdirectory, files in os.walk(os.path.join(folder,subdirectory)):
                for file in files:
                    if ".lab" in file:
                        song_queries.append(file[5:].replace(".lab","").replace("_"," "))
                        loc = os.path.join(root, file)
                        key = open(loc, "r")
                        key = key.read()
                        song_keys.append(key.split("Key\t")[1].split("\n")[0])
     
        
    return tf.convert_to_tensor(song_queries), tf.convert_to_tensor(song_keys)

def get_songs_from_zweieck(folder_name: str):
    #data_dir = pathlib.Path(folder_name)
    #folder = tf.io.gfile.glob(str(data_dir) + '/keylab/The_Beatles')
    folder = os.path.join(folder_name, 'keylab/Zweieck')
    song_queries = []
    song_keys = []
    for root, subdirectories, files in os.walk(folder):
        for subdirectory in subdirectories:
            for root, subdirectory, files in os.walk(os.path.join(folder,subdirectory)):
                for file in files:
                    if ".lab" in file:
                        song_queries.append(file[5:].replace(".lab","").replace("_"," "))
                        loc = os.path.join(root, file)
                        key = open(loc, "r")
                        key = key.read()
                        song_keys.append(key.split("Key\t")[1].split("\n")[0])
     
        
    return tf.convert_to_tensor(song_queries), tf.convert_to_tensor(song_keys)

def get_songs_from_popular_songs(folder_name: str):
    #data_dir = pathlib.Path(folder_name)
    #folder = tf.io.gfile.glob(str(data_dir) + '/keylab/The_Beatles')
    #file = os.path.join(folder_name, 'Rock_D_0_1000.csv')
    file = folder_name
    #key = file.split("PopularSong")[1].split("_")[1]
    data = pd.read_csv(file)
    song_keys = data['Key']
    song_queries = data["Title"]
    song_artist = data["Artist"]
    song_queries = song_artist + ' ' + song_queries
    #song_keys = np.full((len(data)), key)
     
        
    return tf.convert_to_tensor(song_queries), tf.convert_to_tensor(song_keys)

# not yet tested!!!
def get_songs_from_robbie(folder_name: str):
    #data_dir = pathlib.Path(folder_name)
    #folder = tf.io.gfile.glob(str(data_dir) + '/keylab/The_Beatles')
    folder = os.path.join(folder_name, 'keys')
    song_queries = []
    song_keys = []
    for root, subdirectories, files in os.walk(folder):
        for subdirectory in subdirectories:
            for root, subdirectory, files in os.walk(os.path.join(folder,subdirectory)):
                for file in files:
                    if ".txt" in file:
                        song_queries.append(file[3:].replace(".txt","").replace("_"," "))
                        loc = os.path.join(root, file)
                        key = open(loc, "r")
                        key = key.read()
                        song_keys.append(key.split("Key\t")[1].split("\n")[0])
     
        
    return tf.convert_to_tensor(song_queries), tf.convert_to_tensor(song_keys)
