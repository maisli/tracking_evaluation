from track_graph import contract, relabel

def find_moral_edges(edges):
    '''Take all 2D cells between two frames an hypothetical edges between them.
    Filters the edges such that they describe a moral lineage graph. Uses the
    edges weights (higher is better) to assign better edges first.
    '''

    edges.sort(reverse=True)

    prev_nodes = set([p for (w, p, n) in edges])
    next_nodes = set([n for (w, p, n) in edges])

    moral_edges = []

    # greedily match closest continuations
    for (w, pn, nn) in edges:
        if pn in prev_nodes and nn in next_nodes:
            # print("pick %s"%([w, pn, nn]))
            moral_edges.append((pn, nn))
            prev_nodes.remove(pn)
            next_nodes.remove(nn)

    # left over next nodes are splits, assign to closest prev
    for (w, pn, nn) in edges:
        if nn in next_nodes:
            # print("pick %s"%([w, pn, nn]))
            moral_edges.append((pn, nn))
            next_nodes.remove(nn)

    return moral_edges

def greedy_track(cells, edges, edge_weights, threshold):
    '''Simplest possible lineage tracking. Takes a 2D segmentation of cells per
    frame and hypothetical weighted edges between cells of subsequent frames.
    Produces tracks and a track graph by greedily connecting cells morally over
    time.

    Args:

        cells (ndarray):

            2D segmentation per frame.

        edges (ndarray):

            An array of arrays with rows (u, v) per frame-1.

        edge_weights (ndarray):

            An array of 1D arrays with weights w such that edge_weights[z][i] is
            the weight of edges[z][i].

        threshold (float):

            Only edges with a weight larger equal this threshold will be
            considered.
    '''

    filtered_edges = []
    for frame_edges, frame_edge_weights in zip(edges, edge_weights):

        filtered_edges.append([
            (w, u, v)
            for (u, v), w in zip(frame_edges, frame_edge_weights)
            if w >= threshold
        ])

    moral_edges = [ find_moral_edges(f) for f in filtered_edges ]

    track_graph = contract(moral_edges, cells)
    tracks = relabel(cells, track_graph)

    for t in track_graph:
        assert t.label is not None
        assert t.start is not None
        assert t.end is not None, (
            "Track %d has no end, nodes: %s"%(t.label, t.nodes))

    return tracks, track_graph

if __name__ == "__main__":

    import numpy as np

    cells = np.array([
        [[1,1,2,2,2,3,3,]],
        [[4,4,4,5,6,6,7,]],
        [[8,9,9,10,11,12,12,]],
    ])

    edges = np.array([
        [
            [1,4],
            [2,4],
            [2,5],
            [2,6],
            [3,6],
            [3,7],
        ],
        [
            [4,8],
            [4,9],
            [5,10],
            [6,11],
            [6,12],
            [7,12],
        ]])

    edge_weights = np.array([
        [
            0.9,
            0.1,
            0.5,
            0.1,
            0.5,
            0.1,
        ],
        [
            0.9,
            0.8,
            0.9,
            0.7,
            0.8,
            0.1,
        ]])

    tracks, track_graph = greedy_track(cells, edges, edge_weights, 0.5)

    print(cells)
    print(tracks)
    print(track_graph)
