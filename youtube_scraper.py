import csv
import argparse

from youtube_dl import YoutubeDL
from youtube_dl.utils import YoutubeDLError
from nltk.tokenize import RegexpTokenizer
from os.path import exists
import os
import pandas as pd
import time

from dataset_utility import *


class YouTubeDlLogger:
    def __init__(self):
        self.found_problem = False

    def debug(self, msg):
        pass

    def warning(self, msg):
        self.found_problem = True
        print(msg)

    def error(self, msg):
        self.found_problem = True
        print(msg)


# THIS FUNCTION IS FROM: https://github.com/bytedance/GiantMIDI-Piano/blob/930d535a3882f301f7dd8b4c1389072e04989037/dataset.py
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# THIS FUNCTION IS FROM: https://github.com/bytedance/GiantMIDI-Piano/blob/930d535a3882f301f7dd8b4c1389072e04989037/dataset.py
def jaccard_similarity(x, y):
    intersect = intersection(x, y)
    similarity_score = len(intersect) / max(float(len(x)), 1e-8)
    return similarity_score


def similarity(target: str, found: str):
    """
    Calculates the similarity between two strings.

    THIS CODE IS FROM: https://github.com/bytedance/GiantMIDI-Piano/blob/930d535a3882f301f7dd8b4c1389072e04989037/dataset.py

    Parameters
    ----------
    target : str
        The string of the search query for YouTube.
    found : str
        The title of the found YouTube video.

    Returns
    -------
    float
    """
    tokenizer = RegexpTokenizer('[A-Za-z0-9ÇéâêîôûàèùäëïöüÄß]+')

    target_words = tokenizer.tokenize(target.lower())
    found_words = tokenizer.tokenize(found.lower())

    similarity_score = jaccard_similarity(target_words, found_words)

    return similarity_score


def format_filename(name: str):
    """
    Formats the passed string into a format for filenames.

    This means some characters that are not allowed in filenames are replaces with '#' and for
    coding reasons, white spaces are replaced with underscores.

    Parameters
    ----------
    name : str
        The string to format.

    Returns
    -------
    string
    """
    remove_characters = ["/", "\\", "<", ">", ":", "\"", "|", "?", "*"]

    for character in remove_characters:
        name = name.replace(character, "#")
    name = name.replace(" ", "_")

    return name


def search(query: str, nr_results=2):
    """
    Searches YouTube for the passed query, and returns the first 'nr_results' elements.

    Playlist is disabled so that only songs are in the result list and no playlists.
    The Age Limit is set to 17 because videos that are 18+ can only be downloaded with a verified and logged-in
    YouTube account.
    The cache is disabled and also cleared so that no '403 Forbidden Error' is encountered because the cache changed.

    Parameters
    ----------
    query : str
        The title of the YouTube video for which it searches.
    nr_results: int
        The number of returned YouTube search result entries (default=2).

    Returns
    -------
    List<key-value maps>
    """
    logger = YouTubeDlLogger()
    ydl_opts = {'format': 'bestaudio', 'noplaylist': 'True', 'cachedir': False, 'quiet': True, 'age_limit': 17,
                'logger': logger}
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.cache.remove()
            video = ydl.extract_info(f"ytsearch{nr_results}:{query}", download=False)['entries']
        except YoutubeDLError:
            pass
    return None if logger.found_problem else video


def get_song_with_higher_similarity(search_query, video_search_results):
    """
    Iterates over the search results and returns the meta data from the video that is most similar to the search query.

    The similarity is calculated via the Jaccobian Similarity and two cases are considered:
    * only the video title
    * the channel name + the video title
    and the best similarity score of those two is chosen for that video.

    Parameters
    ----------
    search_query : str
        The title of the YouTube video for which was searched.
    video_search_results: List
        A list of meta-data maps from YouTube search results.

    Returns
    -------
    key-value map, int
    """
    highest_similarity_score = 0
    best_video = None

    if video_search_results is None:
        return None, 0

    for video_info in video_search_results:
        similarity_score_1 = similarity(search_query, video_info['title'])
        similarity_score_2 = similarity(search_query, video_info['uploader'] + ' ' + video_info['title'])
        similarity_score = max(similarity_score_1, similarity_score_2)

        # take the shorter video because the longer one is definitely the entire album or a 1-hour version of the song.
        if (similarity_score > highest_similarity_score or (highest_similarity_score >= 0.9 and
                                                           similarity_score >= 0.9 and
                                                           video_info['duration'] < best_video['duration'])) and (video_info['filesize']==None or video_info['filesize']<10000000): # equivalent to 10MB

            highest_similarity_score = similarity_score
            best_video = video_info

    return best_video, highest_similarity_score


def download_video(destination_folder, filename, video_id):
    """
    Downloads the specified video into the specified folder with the stated filename (in mp3 format).

    Parameters
    ----------
    destination_folder : str
        The destination, where to save the audio file.
    filename: str
        The filename for the downloaded audio file.
    video_id: str
        The YouTube video ID from the video to download.

    Returns
    -------
    None
    """
    ydl_opts_dl = {
        'format': 'bestaudio/best',
        'outtmpl': destination_folder + '/' + filename + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    not_done = True
    while(not_done):
        with YoutubeDL(ydl_opts_dl) as ydl:
            try:
                ydl.cache.remove()
                ydl.download(['http://www.youtube.com/watch?v=' + video_id])
                not_done=False
            except:
                # This is needed because sometimes the cache reset doesn't work at first try but on the second.
                # TODO: this is kinda hacky, so find a better fix for this
                print("Trying to download again...")
                time.sleep(2)
                #ydl.cache.remove()
                #ydl.download(['http://www.youtube.com/watch?v=' + video_id])
                #pass


def scrap_youtube(songs_filename: str, destination_folder: str, do_download: bool):
    """
    Extracts info from all songs (and downloads them) from YouTube whose song name is listed in the songs_filename.

    The songs_filename contains the search query and the key of the song separated by a whitespace. And each
    entry is in its own line.

    Parameters
    ----------
    songs_filename : str
        Location of the file containing the search queries (artists + song name + key).
    destination_folder : str
        Location where all downloaded songs and the csv of the similarity scores will be saved.
    do_download : bool
        If True, the mp3 of each song will be downloaded in addition to storing the search result similarities.

    Returns
    -------
    None
    """
    #songs, keys = get_songs_from_popular_songs(songs_filename)
    #songs, keys = get_songs_from_zweieck(songs_filename)
    songs, keys = get_songs_from_queen(songs_filename)
    #songs, keys = get_songs_from_king_carole(songs_filename)
    #songs, keys = get_songs_from_beatles(songs_filename)
    #songs, keys = get_songs_from_keyfinder(songs_filename)  # TODO: make this variable
    #songs, keys = get_songs_from_billboard(songs_filename)
    #songs, keys = get_songs_from_tonality(songs_filename)
    
    if not exists(destination_folder + '/__youtube_similarities.csv'):
        with open(destination_folder + '/' + '__youtube_similarities.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'Similarity Score', 'Key'])
            
    print(exists(destination_folder + '/__youtube_similarities.csv'))
    # to remember where I stopped scraping.
    # TODO: get the offset from the __youtube_similarities.csv as its row-count.
    offset = len(pd.read_csv(destination_folder+'/__youtube_similarities.csv'))

    # TODO: replace the '/' concatenation with e.g. os.path.sep; this is too hard-coded
    with open(destination_folder + '/' + '__youtube_similarities.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for i, song in enumerate(songs[offset:]):
            search_query = song.numpy().decode('utf-8')
            key = keys[i + offset].numpy().decode('utf-8')
            video_info, similarity_score = get_song_with_higher_similarity(search_query, search(search_query))
    
            print(i + 1 + offset, ": ", search_query)

            # store similarity score for later
            filename = format_filename(search_query)
            writer.writerow([filename, similarity_score, key])

            if do_download and similarity_score > 0.6:
                print(f"Found song {video_info['title']} with a similarity score of {similarity_score}")
                download_video(destination_folder, filename, video_info['webpage_url_basename'])

    print("Finished downloading")


def init_parser(parent_directory):
    # Settings
    parser = argparse.ArgumentParser(
        description='Key Signature Estimation Networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--source',      type=str, required=True,
                        help='File containing list of song titles (separated by new lines).')
    parser.add_argument('--destination', type=str, required=True,
                        help='Directory for storing downloaded songs and meta-data file.')

    opt = parser.parse_args()
    s = os.path.join(parent_directory, 'Data/'+opt.source)
    d = os.path.join(parent_directory, 'Data/'+opt.destination)

    if not exists(s) or not exists(d):
        print("ERROR: The passed source file ", opt.source, " or the passed destination ",
              opt.destination, " do not exist.")
        print(s)
        print(d)
    else:
        return opt


if __name__ == '__main__':
    # python youtube_scaper.py --source D:/Praktikum/KeyFinder/__DatasetKeys.txt --destination D:/Praktikum/KeyFinder'
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.split(current_directory)[0]
    arg_opt = init_parser(parent_directory)
    #scrap_youtube(arg_opt.source, arg_opt.destination, True)  # TODO: add the bool 'do_download' also to the argv's.
    source = os.path.join(parent_directory, 'Data/'+str(arg_opt.source))
    destination = os.path.join(parent_directory, 'Data/'+str(arg_opt.destination))
    scrap_youtube(source, destination, True)