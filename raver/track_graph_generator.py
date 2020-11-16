from pprint import pprint
from typing import Iterable, Set

from raver.model.graph.graph import Graph
from raver.model.graph.node import Node
from raver.model.track import Track


class TrackGraphGenerator:

    @classmethod
    def generate_graph(cls, tracks: Iterable[Track]) -> Graph:
        """ Generate a graph with all the given tracks. """
        track_graph = Graph()
        # Dictionary to map tracks to nodes for graph creation
        track_id_to_node = {track.id: Node(track.id) for track in tracks}
        # Avoid unnecessary iterations
        checked_tracks: Set[Track] = set()
        for track in tracks:
            # Doing this here saves us from checking that we are not comparing a track with itself
            checked_tracks.add(track)
            # Compare each track to all others
            for other_track in tracks:
                if other_track in checked_tracks: continue
                # Add an edge to the graph if two tracks are in key
                if track.in_key_with(other_track):
                    track_graph.connect(track_id_to_node[track.id], track_id_to_node[other_track.id])
        return track_graph

if __name__ == '__main__':
    tracks = [
        Track('Jumanji', '4B'),
        Track('Day Of Light', '4A'),
        Track('Sender', '3B'),
        Track('Atlas', '3A'),
        Track('Hear Me Out', '5A'),
        Track('The Space In Between', '4A')
    ]
    graph = TrackGraphGenerator.generate_graph(tracks)

    pprint(graph)