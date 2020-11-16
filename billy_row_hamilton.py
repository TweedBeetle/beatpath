from math import ceil
from typing import Set, List

# from raver.model.graph.graph import Graph
# from raver.model.graph.node import Node

from networkx import Graph
from tqdm import tqdm

from spotify_interface import TrackNode


class WalkStatus:

    def __init__(self, node: TrackNode, visited: Set[TrackNode] = None, walked_path: List[TrackNode] = None):
        self.current_node: TrackNode = node
        self.visited: Set[TrackNode] = visited if visited else set()
        self.walked_path: List[TrackNode] = walked_path if walked_path else list()


class BillyRowHamilton:
    """
    Class designed to solve the William Rowan Hamilton's problem adapted to our current need. The difference is that we
    want the best possible result; that being all the paths that walk through as many nodes as possible without
    repeating vertices, instead of those that walk through every node of the graph.
    """

    def __init__(self):
        self.generator_buffer: List[WalkStatus] = list()

    def find_paths(self, graph: Graph, coverage_proportion: float) -> List[List[TrackNode]]:
        """
        Returns a set of paths that go through at least the `coverage_proportion` of the graph's nodes. If we
        had a graph with 10 nodes, and `coverage_proportion` was 0.7, then we would return all the pseudo hamiltonian
        paths with at least 7 nodes.
        """
        paths: List[List[TrackNode]] = list()
        # Minimal number of nodes that a path should visit to be acceptable
        acceptable_length: int = ceil(len(graph.nodes) * coverage_proportion)
        # Create paths starting from every node
        generator = self.__iteration_generator({WalkStatus(node) for node in graph.nodes})
        bar = tqdm()
        while status := next(generator, None):
            # Last visited node in the current status
            node: TrackNode = status.current_node
            # Already visited nodes
            visited: Set[TrackNode] = status.visited
            # Add current node to the visited nodes
            visited.add(node)
            # Current walked path
            path: List[TrackNode] = status.walked_path
            # Add current node to the end of the path
            path.append(node)
            # Search for path
            # for neighbor in node.edges:
            for neighbor in graph.neighbors(node):
                # Do not walk through nodes that were already visited
                if neighbor in visited:
                    print('skip')
                    continue
                # Add next step to generator buffer
                self.generator_buffer.append(WalkStatus(neighbor, visited.copy(), path[:]))
            # Add path if acceptable
            print(len(path))
            if len(path) >= acceptable_length:
                paths.append(path)
                bar.update()

        return paths

    def __iteration_generator(self, initial_nodes: Set[WalkStatus]):
        """ This allows us to iterate a list that is modified during that iteration. """
        self.generator_buffer = list(initial_nodes)
        index = 0
        # This will yield the next element of the list as long as it has any
        while index < len(self.generator_buffer):
            yield self.generator_buffer[index]
            index += 1
