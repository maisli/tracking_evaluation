from __future__ import print_function
import numpy as np
import h5py
from scipy.ndimage.measurements import center_of_mass

def find_centers(ids):

    all_ids = np.unique(ids)
    coms = center_of_mass(np.ones_like(ids), ids, all_ids)

    return { i: l for i, l in zip(all_ids, coms) if i != 0 }

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def find_possible_edges(ids_prev, ids_next, nodes_prev, nodes_next):

    overlay = np.array([
        ids_prev.flatten(),
        ids_next.flatten(),
        nodes_prev.flatten(),
        nodes_next.flatten()])
    uniques = np.unique(overlay, axis=1)

    possible_edges = {}
    for id_p, id_n, node_p, node_n in zip(uniques[0], uniques[1], uniques[2], uniques[3]):
        if id_p == id_n:
            if id_p not in possible_edges:
                possible_edges[id_p] = []
            possible_edges[id_p].append((node_p, node_n))

    return possible_edges

def find_edges_between(ids_prev, ids_next, nodes_prev, nodes_next):

    edges = []

    possible_edges = find_possible_edges(
        ids_prev,
        ids_next,
        nodes_prev,
        nodes_next)

    # get center of masses of nodes
    locations = find_centers(nodes_prev)
    locations.update(find_centers(nodes_next))

    # print("Locations:")
    # print(locations)

    # for each id
    for i, candidates in possible_edges.iteritems():

        # continuation
        if len(candidates) == 1:

            # print("%d continues"%i)
            edges.append(candidates[0])

        else:

            # print("%d does something complex"%i)
            prev_nodes = set([p for (p, n) in candidates])
            next_nodes = set([n for (p, n) in candidates])

            pairs = []
            for (p, n) in candidates:
                distance = dist(locations[p], locations[n])
                pairs.append((distance, p, n))
            pairs.sort()
            # print("all possible continuations: %s"%pairs)

            # greedily match closest continuations
            for (d, pn, nn) in pairs:
                if pn in prev_nodes and nn in next_nodes:
                    # print("pick %s"%([d, pn, nn]))
                    edges.append((pn, nn))
                    prev_nodes.remove(pn)
                    next_nodes.remove(nn)

            # left over next nodes are splits, assign to closest prev
            for (d, pn, nn) in pairs:
                if nn in next_nodes:
                    # print("pick %s"%([d, pn, nn]))
                    edges.append((pn, nn))
                    next_nodes.remove(nn)

    return edges

def find_edges(ids, nodes):

    print("Finding inter-frame edges...")

    edges = []

    for z in range(ids.shape[0] - 1):
        # print("Searching for edges out of frame ", z)
        edges.append(
            find_edges_between(
                ids[z], ids[z+1], nodes[z], nodes[z+1]))

    return edges

class Track:

    def __init__(self, start, end, label, parent):
        self.start = start
        self.end = end
        self.label = label
        self.parent = parent
        self.nodes = []

    def __repr__(self):

        parent = None
        if self.parent:
            parent = self.parent.label
        return "%d: [%d, %s], nodes %s, parent track: %s"%(self.label,
                self.start, self.end, self.nodes, parent)

def contract(edges, nodes):

    print("Contracting tracks...")

    tracks = []
    node_to_track = {}

    offsprings = {}
    next_offsprings = {}

    # for each frame
    for z in range(len(edges) + 1):

        # print("Contracting in z=%d"%z)

        in_nodes = {}
        out_nodes = {}

        # for all edges leaving the current frame
        if z < len(edges):
            for p, n in edges[z]:
                if p in out_nodes:
                    out_nodes[p].append(n)
                else:
                    out_nodes[p] = [n]
        # for all edges entering the current frame
        if z > 0:
            for p, n in edges[z - 1]:
                if n in in_nodes:
                    in_nodes[n].append(p)
                else:
                    in_nodes[n] = [p]

        # for each node in the current frame
        frame_nodes = list(np.unique(nodes[z]))
        if 0 in frame_nodes:
            frame_nodes.remove(0)
        for node in frame_nodes:
            if node not in in_nodes:
                in_nodes[node] = []
            if node not in out_nodes:
                out_nodes[node] = []

        offsprings = next_offsprings
        next_offsprings = {}

        for node in frame_nodes:

            assert len(in_nodes[node]) <= 1, "Node %d has more than one parents"%node

            if len(in_nodes[node]) == 0 or node in offsprings:

                if node in offsprings:
                    parent = offsprings[node]
                else:
                    parent = None

                track = Track(z, None, node, parent)
                tracks.append(track)

                # print("Start of %s"%track)

            else:

                prev_node = in_nodes[node][0]
                track = node_to_track[prev_node]

                # print("Continuation of %s"%track)

            # now, node has a track, either new or previous
            node_to_track[node] = track
            track.nodes.append(node)

            if len(out_nodes[node]) == 0 or len(out_nodes[node]) > 1:

                track.end = z
                # print("End of track %s, splits into %s"%(track, out_nodes[node]))

                # remember offsprings for processing of next frame
                for out_node in out_nodes[node]:
                    next_offsprings[out_node] = track

    return tracks

def replace(array, old_values, new_values):

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]

def relabel(nodes, tracks):

    print("Relabelling volume...")

    old_values = []
    new_values = []
    for track in tracks:
        for node in track.nodes:
            old_values.append(node)
            new_values.append(track.label)

    old_values = np.array(old_values, dtype=nodes.dtype)
    new_values = np.array(new_values, dtype=nodes.dtype)

    return replace(nodes, old_values, new_values)

def add_track_graph(seg_file):
    '''Add a track graph to an HDF5 file.

    In:

        'volumes/labels/lineages'
        'volumes/labels/cells'

    Out:

        'volumes/labels/tracks'
        'graphs/track_graph'
    '''

    print("Adding track graph to %s..."%seg_file)

    with h5py.File(seg_file, 'r+') as f:

        if 'volumes/labels/tracks' in f:
            del f['volumes/labels/tracks']
        if 'graphs/track_graph' in f:
            del f['graphs/track_graph']

    with h5py.File(seg_file, 'r+') as f:

        lineages = np.array(f['volumes/labels/lineages'])
        cells = np.array(f['volumes/labels/cells'])

        # transform lineages and cells into track graph

        # print("Extracting track graph...")
        edges = find_edges(lineages, cells)
        track_graph = contract(edges, cells)
        tracks = relabel(cells, track_graph)

        for t in track_graph:
            assert t.label is not None
            assert t.start is not None
            assert t.end is not None, (
                "Track %d has no end, nodes: %s"%(t.label, t.nodes))

            # parent = 0 if t.parent is None else t.parent.label
            # print("Track %d from %d to %d, parent %d"%(t.label, t.start, t.end, parent))

        track_graph_data = np.array([
            [
                t.label,
                t.start,
                t.end,
                t.parent.label if t.parent is not None else 0
            ]
            for t in track_graph
        ], dtype=np.uint64)

        # print("Storing track graph...")

        f.create_dataset(
            'volumes/labels/tracks',
            data=tracks,
            compression="gzip")
        f.create_dataset(
            'graphs/track_graph',
            data=track_graph_data,
            compression="gzip")
