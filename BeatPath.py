from datetime import timedelta
from pprint import pprint

from prompt_toolkit import print_formatted_text as fprint, PromptSession
from prompt_toolkit.completion import WordCompleter

import itertools
import random
from prompt_toolkit.validation import Validator
import time
from typing import List

import click
from mcts import mcts
from tqdm import tqdm

from camelot_wheel import CamelotWheel
from spotify_interface import *
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from billy_row_hamilton import BillyRowHamilton
from utils import in_delta, random_sample, get_list_validator


def look_for_longest_path_in_graph(graph: nx.Graph, seconds: float, key_func=len):
    nodes_list = list(graph.nodes)

    if len(nodes_list) == 0:
        return []

    if len(nodes_list) == 1:
        return nodes_list

    paths = []

    total_end = time.time() + seconds

    while time.time() < total_end:

        node_combination_end = time.time() + (seconds / 100)

        start_node = random.choice(nodes_list)
        end_node = random.choice(nodes_list)

        for path in nx.all_simple_paths(graph, start_node, end_node):
            if len(path) == graph.number_of_nodes():
                return path

            paths.append(path)
            if time.time() > node_combination_end:
                break

    # print(paths)
    longest_path = max(paths, key=key_func)

    return longest_path


def path_weight(graph: nx.Graph, path: List[TrackNode]):
    total_weight = 0

    for from_node, to_node in zip(path, path[1:]):
        edge_data = graph.get_edge_data(from_node, to_node)
        total_weight += edge_data['weight'] if edge_data else 1

    return total_weight


def get_path_from_playlist(playlist_id, seconds=1):
    tracks, playlist_length = get_playlist_tracks_and_length(playlist_id)

    track_ids = [track['id'] for track in tracks]

    G = nx.DiGraph()

    camelot_bins = {camelot: set() for camelot in CamelotWheel.all()}

    for node in tqdm(track_nodes(track_ids), total=playlist_length):
        camelot_bins[node.camelot].add(node)
        G.add_node(node)

    bin_sizes = dict(map(lambda t: (t[0], len(t[1])), camelot_bins.items()))
    plt.bar(bin_sizes.keys(), bin_sizes.values())
    plt.show()

    bar = tqdm()

    camelot_graphs = dict()

    for camelot, nodes in camelot_bins.items():

        camelot_graph = nx.DiGraph()

        for from_node in nodes:
            for to_camelot in CamelotWheel.compatibles(camelot):
                for to_node in camelot_bins[to_camelot]:
                    if abs(to_node.tempo - from_node.tempo) <= 10:
                        G.add_edge(from_node, to_node)
                for to_node in nodes:
                    if abs(to_node.tempo - from_node.tempo) <= 10:
                        camelot_graph.add_edge(from_node, to_node)

                    bar.update()

        camelot_graphs[camelot] = camelot_graph

    nonempty_camelots = list(filter(lambda camelot: len(camelot) > 0, camelot_bins.values()))
    num_nonempty_camelots = len(nonempty_camelots)

    end_time = time.time() + seconds
    time_allocation = seconds / (num_nonempty_camelots + 1)

    longest_camelot_paths = {camelot: look_for_longest_path_in_graph(camelot_graphs[camelot], seconds=time_allocation) for camelot in camelot_bins.keys()}

    for camelot, path in longest_camelot_paths.items():
        print(f'{camelot:} {len(path)} out of {len(camelot_bins[camelot])}')

    super_graph = nx.Graph()
    camelot_nodes = dict()

    for camelot, path in longest_camelot_paths.items():

        if len(path) == 0:
            continue

        camelot_node = (camelot, len(path))
        camelot_nodes[camelot] = camelot_node
        super_graph.add_node(camelot_node)

    # empty_camelots = []
    #
    # for camelot, longest_path in longest_camelot_paths.items():
    #     if len(longest_path) == 0:
    #         empty_camelots.append(camelot)
    #
    # for empty_camelot in empty_camelots:
    #     del longest_camelot_paths[empty_camelot]

    for camelot, path_length in super_graph.nodes:
        for neighbour_camelot in CamelotWheel.compatibles(camelot):
            if neighbour_camelot in camelot_nodes:
                super_graph.add_edge(camelot_nodes[camelot], camelot_nodes[neighbour_camelot])

    print()

    time_left = end_time - time.time()
    print(f'time left: {time_left}')

    longest_supergraph_path = look_for_longest_path_in_graph(
        super_graph,
        seconds=time_left,
        key_func=lambda camelot_nodes: sum([node[1] for node in camelot_nodes])
    )

    print(sum([node[1] for node in longest_supergraph_path]))

    beat_path = []

    for camelot, path_length in longest_supergraph_path:
        beat_path += longest_camelot_paths[camelot]

    print(f'found beat path using {len(beat_path)} out of {playlist_length} songs')

    print(beat_path)


def avg(l):
    return sum(l) / len(l)


class BeatPath:  # @todo: minimum dancability and bpm range
    def __init__(self, target_energy_levels, target_set_duration, tempo_range=(110, 140), minimum_danceability=0.6, transition_duration=30, time_limit=1,
                 source_playlist_id=None, seeds=None):
        assert (source_playlist_id is None) != (seeds is None)

        self.target_energy_levels = target_energy_levels
        self.tempo_range = tempo_range
        self.minimum_danceability = minimum_danceability
        self.target_set_duration = target_set_duration

        self.transition_duration = transition_duration

        if seeds is not None:

            recommendations = sp.recommendations(
                seed_artists=seeds['artists'],
                seed_genres=seeds['genres'],
                seed_tracks=seeds['tracks'],
                limit=100
            )['tracks']

            self.source_track_ids = [recommendation['id'] for recommendation in recommendations]
        else:

            self.source_track_ids = get_track_ids_from_playlist(source_playlist_id)

        self.featureses = dict(zip(self.source_track_ids, get_audio_featureses(self.source_track_ids)))
        self.featureses = {track_id: features for track_id, features in zip(self.source_track_ids, get_audio_featureses(self.source_track_ids)) if
                           self.features_are_acceptable(features)}

        # self.analyses = {track_id: get_audio_analysis(track_id) for track_id in self.source_track_ids}

        # self.featureses
        # dict([t for t in self.featureses.items()])

        self.source_track_ids = set(self.featureses.keys())

        print(len(self.source_track_ids))

        self.camelot_bins = {camelot: set() for camelot in CamelotWheel.all()}
        for track_id in self.source_track_ids:
            features = self.featureses[track_id]
            self.camelot_bins[camelot_from_features(features)].add(track_id)

        self.most_common_camelot = max(self.camelot_bins.keys(), key=lambda k: len(self.camelot_bins[k]))
        self.avg_source_track_tempo = avg(list(map(lambda features: features['tempo'], self.featureses.values())))

        tree = mcts(timeLimit=time_limit, explorationConstant=10)
        # tree = mcts(timeLimit=time_limit)
        # tree = mcts(timeLimit=time_limit)
        # tree.search(initialState=PathState(beatpath=self, remaining=self.source_track_ids))
        tree.search(initialState=PathState(beatpath=self, remaining=self.source_track_ids))

        # bestAction = tree.best_action()
        # print(bestAction)

        best_state = tree.best_state()
        self.track_ids = best_state.set_track_ids
        print(list(map(track_name, best_state.set_track_ids)))
        # print(self.set_loss(best_state.track_ids))
        # print(self.set_duration_adjusted(best_state.track_ids))
        print(self.set_duration(best_state.set_track_ids) / 3600)
        print(len(self.track_ids))

    def features_are_acceptable(self, features):
        return all([
            features['danceability'] >= self.minimum_danceability,
            features['tempo'] >= self.tempo_range[0],
            features['tempo'] <= self.tempo_range[1],
            features['duration_ms'] / 1000 >= 3 * self.transition_duration
        ])

    # def last_camelot_added(self):
    #     return camelot_from_features(self.featureses[self.last_track_id_added])
    #
    # def last_tempo_added(self):
    #     return self.tempo(self.last_track_id_added)

    def tempo(self, track_id):
        return self.featureses[track_id]['tempo']

    def key(self, track_id):
        return get_track_key_from_features(self.featureses[track_id])

    def set_is_complete(self, track_ids):
        return self.set_duration(track_ids) > 1.1 * self.target_set_duration

    def set_duration(self, track_ids):
        return sum([self.featureses[track_id]['duration_ms'] for track_id in track_ids]) / 1000

    def set_duration_adjusted(self, track_ids):
        return self.set_duration(track_ids) - self.transition_duration * (len(track_ids))

    def set_loss(self, track_ids):
        # return abs(self.target_set_duration - self.set_duration_adjusted(track_ids))
        # return abs(self.target_set_duration - self.set_duration(track_ids))
        return abs(self.target_set_duration - self.set_duration(track_ids))

    def create_playlist(self, name):
        playlist_from_track_ids(self.track_ids, name)


class PathState:
    def __init__(self, beatpath: BeatPath, track_ids=None, used=None, remaining=None):
        if track_ids is None:
            track_ids = []
        if remaining is None:
            remaining = set()
        if used is None:
            used = set()

        self.beatpath = beatpath

        self.track_ids = track_ids
        self.used = used
        self.remaining = remaining

        self.last_track_for_which_actions_predetermined = None
        self.predetermined_possible_actions = None

    def last_camelot_added(self):
        return camelot_from_features(self.beatpath.featureses[self.track_ids[-1]])

    def last_tempo_added(self):
        return self.beatpath.tempo(self.track_ids[-1])

    def getPossibleActions(self):

        if len(self.track_ids) == 0 or self.track_ids[-1] != self.last_track_for_which_actions_predetermined:

            key_compatible_track_ids = set()

            if len(self.track_ids) == 0:
                return self.beatpath.source_track_ids
                # last_camelot_key = self.beatpath.most_common_camelot
                # last_tempo = self.beatpath.avg_source_track_tempo
            else:
                last_camelot_key = self.last_camelot_added()
                last_tempo = self.last_tempo_added()

            for camelot_key in CamelotWheel.compatibles(last_camelot_key):
                key_compatible_track_ids.update(self.beatpath.camelot_bins[camelot_key].intersection(self.remaining))

            key_compatible_track_ids = filter(lambda track_id: abs(last_tempo - self.beatpath.tempo(track_id)) <= 10, key_compatible_track_ids)
            possible_actions = key_compatible_track_ids
            # todo: energy

            self.predetermined_possible_actions = list(possible_actions) + ['stop']
            if len(self.track_ids) > 0 and abs(self.beatpath.set_duration_adjusted(self.track_ids) - self.beatpath.target_set_duration) <= 60 * 10:
                self.predetermined_possible_actions += ['stop']

            if len(self.track_ids) == 0:
                self.last_track_for_which_actions_predetermined = None
            else:
                self.last_track_for_which_actions_predetermined = self.track_ids[-1]

        # if len(self.track_ids) > 0 and abs(self.beatpath.set_duration_adjusted(self.track_ids) - self.beatpath.target_set_duration) <= 60 * 10:
        #     self.predetermined_possible_actions += ['stop']

        return self.predetermined_possible_actions

    def takeAction(self, action):

        new_track_id = action

        if new_track_id == 'stop':
            return PathState(
                beatpath=self.beatpath,
                track_ids=self.track_ids,
                used=self.used,
                remaining=set()
            )

        new_track_id_set = {new_track_id}

        return PathState(
            beatpath=self.beatpath,
            track_ids=self.track_ids + [new_track_id],
            used=self.used.union(new_track_id_set),
            remaining=self.remaining.difference(new_track_id_set)
        )

    def isTerminal(self):
        return len(self.remaining) == 0 or self.beatpath.set_is_complete(self.track_ids) or len(self.getPossibleActions()) == 0

    def getReward(self):
        # only needed for terminal states
        # return 1 / self.get_loss()
        # return 1 / self.get_loss()
        return self.beatpath.set_duration(self.track_ids)

    def get_loss(self):
        return self.beatpath.set_loss(self.track_ids)

    def __eq__(self, other):
        # raise NotImplementedError()
        return other.set_track_ids == self.track_ids


# @todo filter with liked songs
# @todo: cli

class PathBuilder:
    def __init__(self,
                 set_track_ids,
                 num_recommendations_as_sources=100,
                 vibe_track_ids=None,
                 min_choices=5,
                 num_tracks=30,
                 **kwargs
                 # max_energy_eps=0.05,
                 # max_energy_delta=0.1,
                 # min_energy=0.0,
                 # # max_valence_eps=0.05,
                 # max_valence_delta=0.1,
                 # min_danceability=0.6,
                 # max_tempo_delta=10,
                 ):

        self.params = dict()

        self.simple_attributes = [
            "acousticness",
            "danceability",
            # "duration_ms",
            "energy",
            "instrumentalness",
            # "key",
            "liveness",
            # "loudness",
            # "mode",
            "popularity",
            "speechiness",
            "tempo",
            # "time_signature",
            "valence",
        ]
        for attribute in self.simple_attributes:

            param = f'{attribute}_targets'
            if param in kwargs:
                assert len(kwargs[param]) == num_tracks
                self.params[param] = kwargs[param]
            else:
                self.params[param] = [None] * num_tracks

            for prefix in ["min", "max"]:
                param = prefix + '_' + attribute
                if param in kwargs:
                    self.params[param] = kwargs[param]
                else:
                    if attribute == 'popularity':
                        self.params[param] = 0 if prefix == 'min' else 100
                    elif attribute == 'tempo':
                        self.params[param] = 1 if prefix == 'min' else 300
                    else:
                        self.params[param] = 0 if prefix == 'min' else 1

            for suffix in ["delta", "eps"]:
                param = 'max_' + attribute + '_' + suffix
                if param in kwargs:
                    self.params[param] = kwargs[param]
                else:
                    if attribute == 'popularity':
                        self.params[param] = 100
                    elif attribute == 'tempo':
                        self.params[param] = 300
                    else:
                        self.params[param] = 1

        if set_track_ids is None:
            set_track_ids = []
        assert num_recommendations_as_sources or vibe_track_ids

        self.num_recommendations_as_sources = num_recommendations_as_sources

        if vibe_track_ids is None:
            vibe_track_ids = set()
        self.vibe_track_ids = set(vibe_track_ids)

        # self.target_energy = target_energy
        # self.max_energy_eps = max_energy_eps
        # self.max_energy_delta = max_energy_delta
        # self.min_energy = min_energy
        # self.max_valence_delta = max_valence_delta
        # self.min_danceability = min_danceability
        # self.max_tempo_delta = max_tempo_delta
        self.min_choices = min_choices
        self.num_tracks = num_tracks

        self.set_track_ids = set_track_ids
        self.current_recommendations = set()

        self.set_energy_levels = list(map(get_energy, self.set_track_ids))

    def add_recommendations(self):
        new_recommendation_track_ids = set()

        c = 0

        while len(new_recommendation_track_ids) <= self.num_recommendations_as_sources:

            # seed_tracks = [random.choice(recommendation_track_ids + list(self.source_track_ids))],

            last_track_id = self.set_track_ids[-1]
            next_track_ind = len(self.set_track_ids)

            kwargs = dict()

            for attribute in self.simple_attributes:
                param = f'min_{attribute}'
                kwargs[param] = max(self.params[param], get_feature(last_track_id, attribute) - self.params[f'max_{attribute}_delta'])

                param = f'max_{attribute}'
                kwargs[param] = min(self.params[param], get_feature(last_track_id, attribute) + self.params[f'max_{attribute}_delta'])

                if (target := self.params[f'{attribute}_targets'][next_track_ind]) is not None:
                    kwargs[f'target_{attribute}'] = target

            pprint(kwargs)

            new_recommendations = sp.recommendations(
                seed_tracks=random_sample(self.set_track_ids + list(self.vibe_track_ids), 4) + [last_track_id],
                limit=100,
                **kwargs
            )['tracks']

            # new_recommendations = sp.recommendations(
            #     # seed_tracks=self.set_track_ids + recommendation_track_ids + list(self.source_track_ids),
            #     # seed_tracks=random_sample(self.set_track_ids + recommendation_track_ids + list(self.source_track_ids), 5),
            #     # seed_tracks=random_sample(self.set_track_ids + list(self.vibe_track_ids) + list(self.current_recommendations), 4) + [self.set_track_ids[-1]],
            #     seed_tracks=random_sample(self.set_track_ids + list(self.vibe_track_ids), 4) + [last_track_id],
            #     limit=100,
            #     min_danceability=self.min_danceability,
            #     # min_energy=self.min_energy,
            #     min_energy=max(self.min_energy, get_energy(last_track_id) - self.max_energy_delta),
            #     # max_energy=min(self.max_energy, get_energy(last_track_id) - self.max_energy_delta),
            #     min_tempo=self.tempo_range[0],
            #     max_tempo=self.tempo_range[1],
            #
            # )['tracks']

            new_recommendation_track_ids.update(set([recommendation['id'] for recommendation in new_recommendations]))
            # new_recommendation_track_ids.update(new_recommendation_track_ids)
            # recommendation_track_ids += new_recommendation_track_ids

            if c > 0:
                print(c)
            c += 1

        # self.current_recommendations += recommendation_track_ids
        self.current_recommendations.update(new_recommendation_track_ids)
        self.current_recommendations.update(self.vibe_track_ids)
        # self.source_track_ids.update(recommendation_track_ids)
        # self.vibe_track_ids.update(set(filter(self.features_are_acceptable, self.vibe_track_ids)))

    def set_duration(self):
        milliseconds = sum([get_audio_features(track_id)['duration_ms'] for track_id in self.set_track_ids])
        return timedelta(milliseconds=milliseconds)

    def compatible(self, track_id):  # todo: filter for compatibility again

        # target_energy_level = self.target_energy[len(self.set_track_ids)]

        conditions = [
            track_id not in self.set_track_ids,
            # in_delta(
            #     val=get_energy(track_id),
            #     target=target_energy_level,
            #     delta=self.max_energy_eps
            # ),

        ]
            # + [in_delta(self.params[f''])]

        if len(self.set_track_ids) > 0:
            last_track_id = self.set_track_ids[-1]

            # conditions.append(in_delta(
            #     val=get_energy(track_id),
            #     target=get_energy(last_track_id),
            #     delta=self.max_energy_delta
            # ))
            #
            # conditions.append(in_delta(
            #     val=get_valence(track_id),
            #     target=get_valence(last_track_id),
            #     delta=self.max_valence_delta
            # ))
            #
            # conditions.append(abs(get_tempo(last_track_id) - get_tempo(track_id)) <= self.max_tempo_delta)

            last_camelot = camelot_from_track_id(last_track_id)
            conditions.append(camelot_from_track_id(track_id) in CamelotWheel.compatibles(last_camelot), )

        return all(conditions)

    def run(self, auto):

        self.add_recommendations()

        sesh = PromptSession()

        set_playlist_name = sesh.prompt(
            'How would you like to name your set playlist?\n',
            default=track_name(self.set_track_ids[0]) + ' set' if len(self.set_track_ids) > 0 else '',
            validator=None
        )

        set_playlist_id = create_playlist(set_playlist_name)

        if len(self.set_track_ids) > 0:
            add_tracks_to_playlist(self.set_track_ids, set_playlist_id)

        fprint('Your set playlist has been created!')

        while (current_num_tracks := len(self.set_track_ids)) < self.num_tracks:
            target_energy_level = self.params['energy_targets'][current_num_tracks]

            compatible_track_ids = list(filter(self.compatible, self.current_recommendations))
            compatible_track_ids = sorted(compatible_track_ids, key=get_popularity, reverse=True)

            click.clear()
            fprint(f'Gathered {len(self.current_recommendations)} recommendations.')
            fprint(f'Of those, {len(compatible_track_ids)} are currently compatible')
            if len(compatible_track_ids) < self.min_choices:
                last_camelot = None

                fprint(f'To few choices ({len(compatible_track_ids)} < {self.min_choices}). Adding more...')
                self.add_recommendations()
                continue

                # fprint('<red>no compatible tracks found</red>')
                # if click.confirm('Would you like to add more tracks to the track pool?'):
                #     self.add_recommendations()
                #     continue
                # else:
                #     break
            else:
                last_track_id = self.set_track_ids[-1]
                last_camelot = camelot_from_track_id(last_track_id)

            track_name_dict = {track_name(track_id): track_id for track_id in compatible_track_ids}
            track_names = list(track_name_dict.keys())

            print(f'current set length: {self.set_duration()} ({len(self.set_track_ids)}/{self.num_tracks})')
            print(f'current energy levels:\n{list(map(get_energy, self.set_track_ids))}')
            print(f'current valances levels:\n{list(map(get_valence, self.set_track_ids))}')
            print(f'current keys:\n{list(map(get_track_key, self.set_track_ids))}')
            print(f'the following compatible tracks have been found for key {last_camelot} and energy level {target_energy_level} (energy, key, valence, popularity):')
            # pprint([(track_name(track_id), get_energy_level(track_id)) for track_id in compatible_track_ids])
            print('\n'.join(
                [f'\t{track_name(track_id)} {(get_energy(track_id), get_track_key(track_id), get_valence(track_id), get_popularity(track_id))}' for track_id in
                 compatible_track_ids]))
            playlist_name = '!Options'

            if not auto:

                if playlist_with_name_exists(playlist_name):
                    delete_all_playlists_with_name(playlist_name)

                playlist_from_track_ids([last_track_id] + compatible_track_ids, playlist_name)
                print(f'The playlist "{playlist_name}" has been created for you to browse your options')
                print('')

                validator = Validator.from_callable(
                    lambda track_id: track_id in track_names or track_id == '!',
                    error_message='invalid',
                    move_cursor_to_end=True
                )

                track_name_completer = WordCompleter(track_names)
                selected_track_name = sesh.prompt(
                    'What song would you like to add to your set?\n\t',
                    completer=track_name_completer,
                    validator=validator,
                    complete_while_typing=True,
                )

                if selected_track_name == '!':
                    self.add_recommendations()
                    continue

                selected_track_id = track_name_dict[selected_track_name]

            else:
                selected_track_id = compatible_track_ids[0]

            add_tracks_to_playlist([selected_track_id], set_playlist_id)

            self.set_track_ids.append(selected_track_id)
            self.current_recommendations = set()

    def features_are_acceptable(self, track_id):

        features = get_audio_features(track_id)

        return all([
            features['danceability'] >= self.min_danceability,
            features['tempo'] >= self.tempo_range[0],
            features['tempo'] <= self.tempo_range[1],
            # features['duration_ms'] / 1000 >= 3 * self.transition_duration
        ])


def detour(playlist_id, track_id, playlist_name='!detours'):  # todo: update to implement energy /valence eps / delta and others
    track_id = sp._get_id('track', track_id)
    playlist_track_ids = get_track_ids_from_playlist(playlist_id)

    assert track_id in playlist_track_ids

    adjacent = []

    index = playlist_track_ids.index(track_id)
    if index != 0:
        adjacent.append(playlist_track_ids[index - 1])
    if index != len(playlist_track_ids) - 1:
        adjacent.append(playlist_track_ids[index + 1])

    adjacent_camelots = list(map(camelot_from_track_id, adjacent))

    min_danceability = 0.6  # @todo: min of all tracks in playlist
    tempo_range = [115, 140]  # @todo: bounds of all tracks in playlist

    new_recommendations = sp.recommendations(
        # seed_tracks=self.set_track_ids + recommendation_track_ids + list(self.source_track_ids),
        seed_tracks=random_sample(playlist_track_ids, 5),
        limit=100,
        min_danceability=min_danceability,
        min_tempo=tempo_range[0],
        max_tempo=tempo_range[1],
    )['tracks']

    new_recommendation_track_ids = [recommendation['id'] for recommendation in new_recommendations]

    compatible_track_ids = list(filter(  # @todo: not working
        lambda tid: all([camelot_from_track_id(tid) in CamelotWheel.compatibles(adjacent_camelot) for adjacent_camelot in adjacent_camelots]),
        new_recommendation_track_ids
    ))

    playlist_from_track_ids(compatible_track_ids, playlist_name)


def build():
    baby = [7, 8, 7]
    van_burren = [7, 8, 8, 8, 7, 7, 7, 8, 8, 8, 7, 7, 8, 7, 7]
    skrillex = [8, 8, 8, 8, 7, 7, 9, 9, 8, 8, 7, 8, 7, 7, 4, 8, 8, 7, 7, 8, 7, 8, 7, 7, 8, 8, 7, 7, 7, 8, 4, 6, 7, 8, 6, 6, 8, 8, 7, 7, 6]
    garix = [8, 7, 7, 7, 8, 7, 6, 7, 7, 6, 7, 7, 6, 5, 7, 8, 7, 6, 8, 7, 6, 7, 7, 8, 8, 7]
    harris = [6, 6, 5, 7, 6, 6, 6, 7, 6, 6, 7, 7, 6, 6, 6, 7, 7, 5]
    dash = [5, 7, 7, 7, 8, 7, 7, 7, 7, 7, 8, 8, 7, 7, 8, 7, 7, 7, 8, 7, 8, 8, 8, 7]

    # s1 = [7, 7.5, 8, 8, 7, 7, 8, 9, 8, 7, 8, 7, 8, 8, 9, 8, 7, 6]
    s1 = [7, 7.5, 8, 8, 7, 7, 8, 9, 8, 7, 8, 7, 8, 8, 9, 8, 7, 7, 8, 9, 8, 7, 8, 8, 9, 7]
    pb = PathBuilder(
        # target_energy=list(map(lambda x: x / 10, s1)),
        # max_tempo_delta=10,
        # max_energy_delta=0.1,
        # max_valence_delta=0.1,
        # min_tempo=110,
        # max_tempo=140,
        # min_danceability=0.6,
        set_track_ids=get_track_ids_from_playlist('spotify:playlist:5mR5DzgohQmdbX8qc1zrPX'),
        # vibe_track_ids=get_track_ids_from_playlist('spotify:playlist:0eKWB1M6jEZexdA7yA82Ly'),
        vibe_track_ids=get_track_ids_from_playlist('spotify:playlist:3hLPOzTOfZ6Sv26QqebV3J'),
        # set_track_ids=get_track_ids_from_playlist(get_playlist_id_by_name('who dat')),
        # vibe_track_ids=get_track_ids_from_playlist(get_playlist_id_by_name('techy')),
    )
    pb.run(auto=False)

    # todo: set URI's in cli
    # todo: add new songs to original playlist, not new
    # todo: in auto mode, create !Flag set for trach that should be blacklisted and replaced in current set


if __name__ == '__main__':
    pass
    build()

    # detour('spotify:playlist:5Az6l5T6IsIHXWxtChmE0B', 'spotify:track:7gzOfclimlOrkmeMtpt2GN')
