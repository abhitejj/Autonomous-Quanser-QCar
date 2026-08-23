import heapq
from scipy.interpolate import interp1d
from typing import Dict, Union, Tuple, List, Set

import cv2
import torch
import numpy as np

from hal.utilities.path_planning import RoadMap, RoadMapNode

from .constants import X_OFFSET, Y_OFFSET, ACC_SCALE
from .constants import NODE_POSES_RIGHT_COMMON
from .constants import NODE_POSES_RIGHT_LARGE_MAP
from .constants import EDGE_CONFIGS_RIGHT_COMMON
from .constants import EDGE_CONFIGS_RIGHT_LARGE_MAP


class ACCRoadMap(RoadMap):
    """
    This class is responsible for generating waypoints of the roadmap used in
    the ACC2024 student self-driving competition
    """
    def __init__(self) -> None:
        """
        Initializes the ACCRoadMap object
        """
        # parent class initialization
        super().__init__()
        # read nodes and edges
        node_positions: list = NODE_POSES_RIGHT_COMMON + NODE_POSES_RIGHT_LARGE_MAP
        edges: list = EDGE_CONFIGS_RIGHT_COMMON + EDGE_CONFIGS_RIGHT_LARGE_MAP
        # add scaled nodes to acc map
        for position in node_positions: # [1134, 1454, -HALF_PI]
            position[0] = ACC_SCALE * (position[0] - X_OFFSET)
            position[1] = ACC_SCALE * (Y_OFFSET - position[1])
            self.add_node(position)
        # add scaled edge to acc map
        for edge in edges:
            edge[2] = edge[2] * ACC_SCALE
            self.add_edge(*edge)

    def generate_random_cycle(self, start: int, min_length:int = 3) -> list:
        """
        Generates a random cycle from a given starting node

        Parameters:
        - start: int: The starting node
        - min_length: int: The minimum length of the cycle

        Returns:
        - list: The list of nodes in the cycle
        """
        # depth first search for finding all cycles that start and end at the starting point
        def dfs(start):
            fringe: list = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if path and node == start:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node: RoadMapNode = self.nodes[start]
        cycles: list = [[start_node] + path for path in dfs(start_node) if len(path) > min_length]
        num_cycles: int = len(cycles)
        return cycles[np.random.randint(num_cycles)]

    def generate_path(self, node_sequence: Union[np.ndarray, list]) -> np.array:
        """
        Wraps the generated path as a numpy array object

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - np.array: The path as a numpy array
        """
        if type(node_sequence) == np.ndarray:
            node_sequence = node_sequence.tolist()

        return np.array(super().generate_path(node_sequence)).transpose(1, 0) #[N, (x, y)]

    def generate_path_and_segments(self, node_sequence: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wraps the generated path and segments as a numpy array object

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The path and segments as numpy arrays
        """
        if type(node_sequence) == np.ndarray:
            node_sequence = node_sequence.tolist()

        # convert from map node to index
        sequence_ids = node_sequence
        if not type(sequence_ids[0]) == int:
            sequence_ids = [node.index for node in sequence_ids]

        # generate path and find which waypoints belong to which road segment
        path = []
        segments = {}
        waypoint_index = 0
        for i in range(1, len(sequence_ids)):
            sub_sequence = sequence_ids[i-1:i+1]
            sub_sequence_path = np.array(super().generate_path(sub_sequence)).transpose(1, 0) #[N, (x, y)]
            sub_sequence_length = sub_sequence_path.shape[0]

            path.append(sub_sequence_path)
            segments[(waypoint_index, waypoint_index + sub_sequence_length - 1)] = np.array(sub_sequence)
            waypoint_index += sub_sequence_length

        path = np.vstack(path)
        return path, segments

    def prepare_map_info(self, node_sequence: list) -> Tuple[dict, np.ndarray]:
        """
        Provide the position informations related to the node sequence

        Parameters:
        - node_sequence: Union[np.ndarray, list]: The sequence of nodes

        Returns:
        - Tuple[dict, np.ndarray]: The list of nodes' and waypoints' position
        """
        node_dict: Dict[str, np.ndarray] = {}
        for node_id in node_sequence:
            pose: np.ndarray = self.nodes[node_id].pose
            node_dict[node_id] = pose # x, y, angle

        waypoint_sequence = self.generate_path(node_sequence)
        return node_dict, waypoint_sequence
    

class CustomRoadMapNode(RoadMapNode):
    """Class for representing nodes in the graph of a RoadMap

    Attributes:
        pose (numpy.ndarray): Node's pose in the form [x, y, th].
        inEdges (list): List of incoming edges.
        outEdges (list): List of outgoing edges.
    """

    def __init__(self, pose: np.ndarray, index: int):
        """Initialize a RoadMapNode instance.

        Args:
            pose (list or numpy.ndarray): Node's pose in the form [x, y, th].
        """
        assert len(pose) == 3, "Pose must be in the form of [x, y, th]"

        self.pose = np.array(pose).reshape(3, 1)
        self.index = index
        self.inEdges = []
        self.outEdges = []


class CustomRoadMap(RoadMap):
    def add_node(self, pose, index):
        """
        Add a node to the roadmap.

        Args:
            pose (list or numpy.ndarray): Node's pose in the form [x, y, th].
        """
        self.nodes.append(CustomRoadMapNode(pose, index))

    def generate_path(self, nodeSequence):
        """
        Generate the shortest path passing through the given sequence of nodes

        Args:
            nodeSequence (list or tuple): Sequence of node indices.

        Returns:
            numpy.ndarray: generated path as a 2xn numpy array
        """
        assert isinstance(nodeSequence, (list, tuple)), \
            "Node sequence must be provided as either a list or a tuple."

        path = np.empty((2, 0))
        for i in range(len(nodeSequence) - 1):
            pathSegment, _ = self.find_shortest_path(
                nodeSequence[i],
                nodeSequence[i+1]
            )
            if pathSegment is None:
                return None
            path = np.hstack((path, pathSegment[:, :-1]))
        return path

    def find_shortest_path(self, startNode, goalNode):
        """Find the shortest path between two nodes using the A* algorithm.

        Args:
            startNode (int or RoadMapNode): Starting node (index or instance).
            goalNode (int or RoadMapNode): Goal node (index or instance).
            radius (float): Minimum turning radius.

        Returns:
            path: generated path as a 2xn numpy array
        """
        if startNode == goalNode:
            return None

        if type(startNode) == int:
            startNode = self.nodes[startNode]
        if type(goalNode) == int:
            goalNode = self.nodes[goalNode]

        # Initialize the open set and closed set
        openSet = []
        closedSet = set()

        # Add the start node to the open set with a cost of 0 and an
        # f-score equal to the heuristic estimate
        heapq.heappush(
            openSet,
            (0 + self._heuristic(startNode, goalNode), startNode)
        )

        # Initialize the g-scores for each node to infinity
        gScore = {node: float('inf') for node in self.nodes}
        gScore[startNode] = 0

        # Initialize the 'came from' (node, edge) pair
        # for each node to None
        cameFrom = {node: None for node in self.nodes}

        while openSet:
            # Pop the node with the lowest f-score from the open set
            currentNode = heapq.heappop(openSet)[1]

            if currentNode == goalNode:
                # Goal node found, construct the optimal path, then return
                path = goalNode.pose[:2,:]
                node = goalNode
                index_path = np.array([node.index])
                while True:
                    (node, edge) = cameFrom[node]
                    index_path = np.concatenate([index_path, [node.index]])
                    path = np.hstack((
                        node.pose[:2,:],
                        edge.waypoints,
                        path
                    ))
                    if cameFrom[node] is None:
                        break

                return path, index_path

            closedSet.add(currentNode)

            for edge in currentNode.outEdges:
                neighborNode = edge.toNode
                if neighborNode in closedSet:
                    # Neighbor node already explored
                    continue

                if edge.length is None:
                    tentative_g = float('inf')
                else:
                    tentative_g = gScore[currentNode] + edge.length

                if tentative_g < gScore[neighborNode]:
                    # New path to neighbor node found,
                    # update parent pointer and g-score
                    cameFrom[neighborNode] = (currentNode, edge)
                    gScore[neighborNode] = tentative_g

                    # Add the neighbor node to the open set with a cost
                    # equal to the g-score plus the heuristic estimate
                    hScore = self._heuristic(neighborNode, goalNode)
                    heapq.heappush(
                        openSet,
                        (gScore[neighborNode] + hScore, neighborNode)
                    )

        # Open set is empty and goal node not found, no path exists
        return None


class ACCRoadMap2(CustomRoadMap): 
    def __init__(
        self,
        n_mp_pl: int = 1024,
        n_mp_pl_node: int = 30,
        removed_node_list: list = [],
    ): 
        # parent class initialization 
        super().__init__()
        # read nodes and edges 
        node_positions = NODE_POSES_RIGHT_COMMON + NODE_POSES_RIGHT_LARGE_MAP 
        edges = EDGE_CONFIGS_RIGHT_COMMON + EDGE_CONFIGS_RIGHT_LARGE_MAP  
        # add scaled nodes to acc map 
        for index, position in enumerate(node_positions): 
            position[0] = ACC_SCALE * (position[0] - X_OFFSET) 
            position[1] = ACC_SCALE * (Y_OFFSET - position[1]) 
            self.add_node(position, index) 
        # add scaled edge to acc map 
        for edge in edges: 
            edge[2] = edge[2] * ACC_SCALE 
            self.add_edge(*edge)

        #create polylines for rendering
        self.agent_length = 0.4
        self.agent_width = 0.2
        self.road_width = 0.27
        self.removed_node_list = removed_node_list

        self.map_polylines = []

        # each edge waypoint is spaced 1 cm apart, so if n_mp_pl_node = 30 then
        # each polyline represents a 30cm road segment (a bit smaller than the)
        # size of the qcar
        self.n_mp_pl = n_mp_pl #number of polylines in the map
        self.n_mp_pl_node = n_mp_pl_node #number of nodes in a polyline
        self.map_valid = np.zeros([self.n_mp_pl, self.n_mp_pl_node], dtype=bool)
        self.map_id = np.zeros([self.n_mp_pl], dtype=np.int64) - 1
        self.map_pos = np.zeros([self.n_mp_pl, self.n_mp_pl_node, 2], dtype=np.float32)
        self.map_dir = np.zeros([self.n_mp_pl, self.n_mp_pl_node, 2], dtype=np.float32)

        #smart map format, treat all as road edge for qlabs TODO: add crosswalk
        self.map_infos = {
            #"lanes": [],
            "road_edge": [],
            #"road_lines": [],
            #"crosswalk": []
        }
        
        self.map_counter = 0 #n_mp
        self.point_count = 0
        for i, edge in enumerate(self.edges):
            if edge.waypoints is None:
                continue

            waypoints = np.transpose(edge.waypoints) #pl_pos
            dw = np.diff(waypoints, axis=0) #pl_dir
            theta = np.arctan2(dw[:, 1], dw[:, 0])
            theta = np.concatenate([[theta[0]], theta])
            right_lane = waypoints + np.transpose(np.array([np.cos(theta + (np.pi/2)), np.sin(theta + (np.pi/2))]))*(self.road_width / 2)
            left_lane = waypoints + np.transpose(np.array([np.cos(theta - (np.pi/2)), np.sin(theta - (np.pi/2))]))*(self.road_width / 2)

            #segment polylines (Trafficbots)
            polyline_len = dw.shape[0]
            polyline_cuts = np.linspace(0, polyline_len, polyline_len // self.n_mp_pl_node + 1, dtype=int, endpoint=False)
            num_cuts = len(polyline_cuts)
            for idx_cut in range(num_cuts):
                idx_start = polyline_cuts[idx_cut]
                if idx_cut + 1 == num_cuts:
                    #last cut
                    idx_end = polyline_len
                else:
                    idx_end = polyline_cuts[idx_cut + 1]

                #note: there is a left and right road line for each edge, so we define 2 polylines for each edge
                self.map_valid[self.map_counter:self.map_counter+2, : idx_end - idx_start] = True
                self.map_pos[self.map_counter, : idx_end - idx_start] = right_lane[idx_start:idx_end]
                self.map_pos[self.map_counter + 1, : idx_end - idx_start] = left_lane[idx_start:idx_end]
                self.map_dir[self.map_counter:self.map_counter+2, : idx_end - idx_start] = dw[idx_start:idx_end]
                #self.map_id[self.map_counter:self.map_counter+2] = edge.fromNode.index
                #self.map_counter += 2

                #get polyline info (SMART)
                cur_info_right = {
                    "id": self.map_counter,
                    "polyline_index": (self.point_count, self.point_count + right_lane[idx_start:idx_end].shape[0])
                }
                self.point_count += right_lane[idx_start:idx_end].shape[0]

                cur_info_left = {
                    "id": self.map_counter + 1,
                    "polyline_index": (self.point_count, self.point_count + left_lane[idx_start:idx_end].shape[0])
                }
                self.point_count += left_lane[idx_start:idx_end].shape[0]

                self.map_infos["road_edge"].append(cur_info_right)
                self.map_infos["road_edge"].append(cur_info_left)
                self.map_polylines.append(right_lane)
                self.map_polylines.append(left_lane)
                self.map_counter += 2

        #concatenate polylines
        self.map_polylines = np.concatenate(self.map_polylines, axis=0).astype(np.float32)

        #parse smart map features
        polygon_ids = [x["id"] for x in self.map_infos["road_edge"]]
        num_polygons = len(polygon_ids) #should be the same as map_counter
        point_position = [None] * num_polygons

        for _seg in self.map_infos["road_edge"]:
            _idx = polygon_ids.index(_seg["id"])
            roadline = self.map_polylines[_seg["polyline_index"][0] : _seg["polyline_index"][1]]
            point_position[_idx] = roadline[:-1, :2]

        num_points = np.array([point.shape[0] for point in point_position], dtype=int)
        point_to_polygon_edge_index = np.stack([
            np.arange(num_points.sum(), dtype=int),
            np.arange(num_polygons, dtype=int).repeat(num_points),
        ], axis=0)

        point_position = np.concatenate(point_position, axis=0) #equivalent to map_data["map_point"]["position"]

        # get the full map boundary for rendering
        pos = self.map_pos[self.map_valid]
        xmin = pos[:, 0].min()
        ymin = pos[:, 1].min()
        xmax = pos[:, 0].max()
        ymax = pos[:, 1].max()
        self.map_boundary = np.array([xmin, xmax, ymin, ymax])
    
    #generate a random cycle from a given starting node
    def generate_random_cycle(self, start, min_length=3, max_length=15):
        #depth first search for finding all cycles that start and end at the starting point
        if start in self.removed_node_list:
            pass
        def dfs(start):
            fringe = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if path and node == start:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node = self.nodes[start]
        cycles = [[start_node] + path for path in dfs(start_node) if min_length <= len(path) <= max_length]
        num_cycles = len(cycles)

        return cycles[np.random.randint(num_cycles)]

    #generate a random cycle from a given starting node
    def generate_random_path(self, start, length=10):
        #depth first search for finding all cycles that start and end at the starting point
        def dfs(start):
            fringe = [(start, [])]

            while fringe:
                node, path = fringe.pop()
                if len(path) == length:
                    yield path
                    continue
                for next_edges in node.outEdges:
                    next_node = next_edges.toNode
                    if next_node in path:
                        continue
                    fringe.append((next_node, path + [next_node]))

        start_node = self.nodes[start]
        paths = [[start_node] + path for path in dfs(start_node)]
        num_paths = len(paths)

        try:
            out = paths[np.random.randint(num_paths)]
        except:
            breakpoint()

        return out

    #wrap as numpy array object
    def generate_path(self, sequence):
        if type(sequence) == np.ndarray:
            sequence = sequence.tolist()

        #convert from map node to index
        sequence_ids = sequence
        if isinstance(sequence_ids[0], RoadMapNode):
            sequence_ids = [node.index for node in sequence_ids]

        #generate path and find which waypoints belong to which road segment
        path = []
        segments = {}
        waypoint_index = 0
        for i in range(1, len(sequence_ids)):
            sub_sequence = sequence_ids[i-1:i+1]
            sub_sequence_path = super().generate_path(sub_sequence)
            sub_sequence_path = np.array(sub_sequence_path).transpose(1, 0) #[N, (x, y)]
            sub_sequence_length = sub_sequence_path.shape[0]

            path.append(sub_sequence_path)
            segments[(waypoint_index, waypoint_index + sub_sequence_length - 1)] = np.array(sub_sequence)
            waypoint_index += sub_sequence_length

        path = np.vstack(path)
        return path, segments
