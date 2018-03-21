from __future__ import print_function
import sys
import h5py
from track_graph import add_track_graph

gt_files = [
    '/groups/kainmueller/home/maisl/evaluation/data/pro01.hdf',
    '/groups/kainmueller/home/maisl/evaluation/data/pro02.hdf',
    '/groups/kainmueller/home/maisl/evaluation/data/pro03.hdf',
    '/groups/kainmueller/home/maisl/evaluation/data/pro04.hdf',
    '/groups/kainmueller/home/maisl/evaluation/data/pro05.hdf',
    '/groups/kainmueller/home/maisl/evaluation/data/per01.hdf',
    '/groups/kainmueller/home/maisl/evaluation/data/per02.hdf',
    '/groups/kainmueller/home/maisl/evaluation/data/per03.hdf',
]

for gt_file in gt_files:

    with h5py.File(gt_file, 'r') as f:
        track_graph_present = 'graphs/track_graph' in f

    if not track_graph_present:

        print("Adding GT track graph to ", gt_file)
        add_track_graph(gt_file)

    else:

        print("Skipping ", gt_file)
