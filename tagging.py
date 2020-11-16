import os

from mutagen.easyid3 import EasyID3 as id3

# audio = id("example.mp3")
# audio['title'] = u"Example Title"
# audio['artist'] = u"Me"
# audio['album'] = u"My album"
# audio['composer'] = u"" # clear
# audio.save()
from mutagen.id3 import ID3NoHeaderError
from tqdm import tqdm

from spotify_interface import get_genres_by_track_info

music_file_suffixes = ['mp3', 'wav', 'flac']

suffixes = set()
c = 0


class Counter:
    def __init__(self):
        self.count = 0

    def incr(self, *args, **kwargs):
        self.count += 1


def true(*args, **kwargs):
    return True


def artists_filter(path, dirs, file):
    suffix = file_extension(file)
    if suffix.lower() in music_file_suffixes:
        split_path = path.split('\\')
        return split_path[-3] == 'Artists'

    return False


def file_extension(file):
    return file.split('.')[-1]


def walk(dir, callback, filter=true, show_progress=True):
    bar = None

    if show_progress:
        counter = Counter()
        walk(dir, callback=counter.incr, filter=filter, show_progress=False)

        num_files = counter.count

        bar = tqdm(total=num_files)

    for path, dirs, files in os.walk(dir):
        for file in files:
            if filter(path, dirs, file):
                callback(path, dirs, file)
                if bar is not None:
                    bar.update()

# @todo remove all 5 and 1 star ratings

def tag_keys(path, dirs, file):
    split_path = path.split('\\')

    location = os.path.join(*(split_path + [file]))

    try:
        tags = id3(location)
    except ID3NoHeaderError:
        tags = id3()

    return tags.keys()


def add_simple_tags_callback(path, dirs, file):
    split_path = path.split('\\')
    artist = split_path[-2]
    album = split_path[-1]
    title = file.split('-')[0].strip()

    location = os.path.join(*(split_path + [file]))
    try:
        tags = id3(location)
    except ID3NoHeaderError:
        tags = id3()

    if 'title' not in tags.keys():
        tags['title'] = title

    if 'artist' not in tags.keys():
        tags['artist'] = artist

    if 'album' not in tags.keys():
        tags['album'] = album

    if 'genre' not in tags.keys():
        genres = get_genres_by_track_info(title=title, artist=artist, album=album)

        if genres is not None:
            # tags['genre'] = ', '.join(genres)
            tags['genre'] = genres
        # else:
        #     print('no genres found')

    tags.save(location)


def add_tags(dir='F:\Media\Audible\Music\Artists'):
    walk(dir, callback=add_simple_tags_callback, filter=artists_filter, show_progress=True)


if __name__ == '__main__':
    pass
    add_tags()

    #@todo find songs already downloaded and remove them from dl playlist

    # tags = set()
    # walk(
    #     'F:\Media\Audible\Music\Artists',
    #     callback=lambda path, dirs, file: tags.update(tag_keys(path, dirs, file)),
    #     filter=artists_filter
    # )
    # print(tags)
