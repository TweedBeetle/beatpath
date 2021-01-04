import time
from pprint import pformat

from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import spotipy
import os

import creds
from cachetools import cached, LRUCache

from utils import list_cached, execute_in_chunks, PersistentLRUCache, chunks

os.environ['SPOTIPY_CLIENT_ID'] = creds.spotify_client_id
os.environ['SPOTIPY_CLIENT_SECRET'] = creds.spotify_client_secret

KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODES = ['Minor', 'Major']  # @todo: move to enums.py

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

_user_sp = None


def get_user_sp():
    global _user_sp

    if _user_sp is None:
        scope = 'playlist-read-collaborative, playlist-modify-public, playlist-read-private, playlist-modify-private, user-read-private'
        _user_sp = spotipy.Spotify(auth_manager=SpotifyOAuth(cache_path='cache/spotipy.cache', redirect_uri='https://example.com/callback/', scope=scope))

    return _user_sp


class TrackNode:
    def __init__(self, track_id, features=None):
        self.track_id = track_id

        self.features = sp.audio_features(self.track_id)[0] if features is None else features

        self.key = get_track_key_from_features(self.features)
        self.camelot = camelot_from_key(self.key)

        self.tempo = self.features['tempo']

        self.name = track_name(self.track_id)


# @cached(PersistentLRUCache(location='cache/audio_features', maxsize=10000), key=id)
# @list_cached(PersistentLRUCache(location='cache/audio_featureses', maxsize=10000), key=id)
@list_cached(LRUCache(maxsize=10000), key=id)
@execute_in_chunks()
def get_audio_featureses(track_ids):
    return sp.audio_features(track_ids)


@list_cached(LRUCache(maxsize=10000), key=id)
@execute_in_chunks()
def get_tracks_data(track_ids):
    tracks = sp.tracks(track_ids)
    # print(tracks)
    return tracks['tracks']


# @list_cached(PersistentLRUCache(location='cache/audio_features', maxsize=10000), key=id)
# @list_cached(LRUCache(maxsize=10000), key=id)
def get_audio_features(track_id):
    return get_audio_featureses([track_id])[0]


def get_track_data(track_id):
    return get_tracks_data([track_id])[0]


def get_energy(track_id):
    return get_audio_features(track_id)['energy']


def get_valence(track_id):
    return get_audio_features(track_id)['valence']


def get_popularity(track_id):
    return get_track_data(track_id)['popularity']


def get_feature(track_id, feature):
    if feature in ['popularity']:
        return get_track_data(track_id)[feature]
    else:
        # print(get_audio_features(track_id).keys())
        return get_audio_features(track_id)[feature]


# @list_cached(LRUCache(maxsize=10000), key=id)
# def get_audio_features(track_ids):
#     execute_in_chunks(sp.audio_features, track_ids)
#
#     if isinstance(track_ids, str):
#         return sp.audio_features(track_ids)[0]
#
#     features = []
#
#     for chunk in chunks(track_ids, 99):
#         features += sp.audio_features(chunk)
#
#     return features

# @cached(PersistentLRUCache(location='cache/audio_features', maxsize=1000))
@cached(LRUCache(maxsize=1000))
def get_audio_analysis(track_id):
    return sp.audio_analysis(track_id)


def get_audio_analyses(track_ids):
    return map(sp.audio_analysis, track_ids)


def track_nodes(track_ids):
    featureses = get_audio_featureses(track_ids)

    return [TrackNode(track_id, features) for track_id, features in zip(track_ids, featureses)]


# scope = 'playlist-read-collaborative, playlist-modify-public, playlist-read-private, playlist-modify-private'
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(cache_path='cache/spotipy.cache', redirect_uri='https://example.com/callback/', scope=scope))
# current_user = sp.me()
# sp.user_playlist_create(user=current_user['id'], name='aaaatestp')
#


def playlist_from_track_ids(track_ids, name):
    playlist_id = create_playlist(name)
    add_tracks_to_playlist(track_ids, playlist_id=playlist_id)


def create_playlist(name):
    user_sp = get_user_sp()
    current_user = user_sp.me()

    playlist = user_sp.user_playlist_create(user=current_user['id'], name=name)
    return playlist['id']


def add_tracks_to_playlist(track_ids, playlist_id):
    user_sp = get_user_sp()
    current_user = user_sp.me()

    for chunk in chunks(track_ids, 1):
        user_sp.playlist_add_items(
            playlist_id=playlist_id,
            items=chunk,
            # user=current_user['id'],
        )
        # time.sleep(1.01)


def get_user_playlists(limit=50):  # TODO: fix so that i get all playlists when more than 50 exist
    playlists = get_user_sp().current_user_playlists(limit=limit)['items']
    # print(playlists)
    return playlists


def playlist_with_name_exists(name):
    return any([playlist['name'] == name for playlist in get_user_playlists()])


def delete_all_playlists_with_name(name, max=1):
    user_sp = get_user_sp()
    current_user = user_sp.me()

    playlists_to_delete = list(filter(lambda playlist: playlist['name'] == name, get_user_playlists()))
    if len(playlists_to_delete) > max:
        print(f'to many playlists:\n{pformat(playlists_to_delete)}')
        return

    for playlist in playlists_to_delete:
        user_sp.current_user_unfollow_playlist(playlist['id'])

    # for playlist in get_user_playlists():
    #     if playlist['name'] == name:
    #         print(f'deleting {playlist}')
    #         # user_sp.current_user_unfollow_playlist(playlist)


def get_playlist_id_by_name(name):
    playlists = list(filter(lambda playlist: playlist['name'] == name, get_user_playlists()))

    if len(playlists) == 0:
        raise RuntimeError(f'no playlist "{name}" found')

    if len(playlists) > 1:
        print(f'multiple playlists "{name}" found. using first')

    return playlists[0]['uri']


def delete_playlist_by_name_if_present(playlist_name):
    pass  # @todo


@cached(LRUCache(maxsize=1000))
def track_name(track_id):
    return sp.track(track_id)['name']


# track_name('spotify:track:1jO5cLkg1lJdYqgR2WPAwM')

from tinytag import TinyTag


def get_bitrate(fname):
    tag = TinyTag.get(fname)
    return tag.bitrate


def get_tracks_from_playlist(playlist_id):  # todo: only returns 100. why? -> fix
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return [track['track'] for track in tracks]


def get_track_ids_from_playlist(playlist_id):
    return [track['id'] for track in get_tracks_from_playlist(playlist_id)]


def get_playlist_length(playlist_id):
    return sp.playlist(playlist_id)['tracks']['total']


def get_playlist_tracks_and_length(playlist_id):
    playlist = sp.playlist(playlist_id)

    length = sp.playlist(playlist_id)['tracks']['total']
    tracks = map(lambda item: item['track'], playlist['tracks']['items'])
    return tracks, length


def key_and_mode_ind_to_musical_key(key_ind, mode_ind):
    musical_key = f'{KEYS[key_ind]} {MODES[mode_ind]}'

    return musical_key


def get_track_key(track_id):
    analysis = sp.audio_analysis(track_id)
    key_ind = analysis['track']['key']
    mode_ind = analysis['track']['mode']

    return key_and_mode_ind_to_musical_key(key_ind, mode_ind)


def get_track_key_from_features(features):
    key_ind = features['key']
    mode_ind = features['mode']

    return key_and_mode_ind_to_musical_key(key_ind, mode_ind)


def get_tempo(track_id):
    return get_audio_features(track_id)['tempo']


def camelot_from_key(key):
    """http://www.harmonic-mixing.com/howto.aspx"""

    key_ind = KEYS.index(key.split()[0])

    mode = key.split()[1]

    start = 5 if mode == 'Minor' else 8

    key_designator = str((start + key_ind * 7) % 12)

    if key_designator == '0':
        key_designator = '12'

    mode_designator = ['A', 'B'][MODES.index(mode)]

    return key_designator + mode_designator


def camelot_from_features(features):
    return camelot_from_key(get_track_key_from_features(features))


def camelot_from_track_id(track_id):
    return camelot_from_key(get_track_key_from_features(get_audio_features(track_id)))


def get_track_camelot(track_id):
    return camelot_from_key(get_track_key(track_id))


def get_track_id_by_file(location):
    tag = TinyTag.get(location)
    # tag. artist? tag.name? etc


def get_track_by_info(title=None, artist=None, album=None):
    assert not (title is None and artist is None and album is None)

    parts = []

    if title is not None:
        parts.append(f'track:{title}')
    if artist is not None:
        parts.append(f'artist:{artist}')
    if album is not None:
        parts.append(f'album:{album}')

    q = ' '.join(parts)
    response = sp.search(q, type='track')

    items = response['tracks']['items']

    if len(items) == 0:
        return None
    return items[0]


def get_genres(track_id):
    artist_id = sp.track(track_id=track_id)['artists'][0]['id']
    return sp.artist(artist_id)['genres']


# @todo: backup all my spotify playlists


@cached(LRUCache(maxsize=1000))
def get_genres_by_track_info(*args, **kwargs):
    track = get_track_by_info(*args, **kwargs)

    if track is None:
        return None

    track_id = track['id']
    return get_genres(track_id)


if __name__ == '__main__':
    pass

    # playlist_from_track_ids(['spotify:track:3ifkXomA0L2CmqSWyrVPkm'], '000test')

    get_user_sp()

    tids = get_track_ids_from_playlist('spotify:playlist:46GbJCgS1j3YiWTfTs7kCy')

    pass

    # get_energy_level(tids[0])
    print(get_energy('spotify:track:7mA03cPnG3UkLgf1ed87fI'))

    # fs = get_audio_features(tids)
    # fss = sorted(fs, key=lambda f: f['danceability'])
    #
    # print(get_audio_features('spotify:track:0Qn0i8df7Q76ej  3RAXAtI2'))

    # get_track_by_info(title='sandstorm', artist='darude')
    # get_track_by_infso(title='Hey Brother', artist='Avicii')
    # get_track_by_info(title=None, artist='Avicii', album='true')

    # track_features_and_analysis()
    # print(get_track_key('spotify:track:3ifkXomA0L2CmqSWyrVPkm'))
    # print(camelot_from_key('D Major'))

    # tid = 'spotify:track:6cx06DFPPHchuUAcTxznu9'

    # tracks_from_playlist()

    # tid = 'spotify:track:1jO5cLkg1lJdYqgR2WPAwM'
    # features = sp.audio_features(tid)[0]
    # analysis = get_audio_analysis(tid)
    # print()
    #
    # print(get_track_key(tid))
    # print(get_track_camelot(tid))
