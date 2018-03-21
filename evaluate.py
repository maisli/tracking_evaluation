import sys
import h5py
import numpy as np
import json
from segtra import evaluate_segtra
from track_graph import add_track_graph
from postprocess import *
from greedy_track import greedy_track
from scipy import ndimage
from tra import get_tra
import argparse


def parse_arguments():

    print 'parsing arguments...'

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--res', help='result file name', type=str,
            default=None, dest='res_file')
    parser.add_argument('-g', '--gt', help='gt file name', type=str,
            default=None, dest='gt_file')
    parser.add_argument('-o', '--output', help='json output file name', type=str,
            default=None, dest='output_file')
    parser.add_argument('-err', '--output_errors', help='output errors',
            action='store_true', default=False, dest='output_errors')
    parser.add_argument('-p', '--process', 
            help='how to postprocess data: dilate, dilate3d, watershed, watershed3d',
            nargs='+', default=['dilate'])
    parser.add_argument('-i', '--ignore_px', help='ignore pixel smaller and equal than this',
            type=int, default=3)
    parser.add_argument('--recreate_track', help='recreate unique cells and track graph',
            action='store_true', default=False, dest='recreate_track')

    args = parser.parse_args()

    return args


def evaluate_files(res_file, gt_file, output_file=None, output_errors=False, process=[], 
        ignore_px=0, recreate_track=False):


    print 'reading gt volume...'

    with h5py.File(gt_file, 'r') as f:
        mask = np.array(f['volumes/labels/ignore'])
        raw = np.array(f['volumes/raw'])
        gt_tracks = np.array(f['volumes/labels/tracks'])
        gt_track_graph = np.array(f['graphs/track_graph'])
    
    print 'reading volumes...'
    
    with h5py.File(res_file, 'r+') as f:
        
        if recreate_track: 
            
            if 'volumes/labels/tracks' in f:
                print 'delete tracks'
                del f['volumes/labels/tracks']
            if 'graphs/track_graph' in f:
                print 'delete graph'
                del f['graphs/track_graph']
            if 'volumes/labels/unique_ids' in f:
                print 'delete unique_ids'
                del f['volumes/labels/unique_ids']
    
        if 'volumes/labels/vertex_errors' in f:
            print 'delete vertex errors'
            del f['volumes/labels/vertex_errors']
        if 'volumes/labels/edge_errors' in f:
            print 'delete edge errors'
            del f['volumes/labels/edge_errors']

        track_graph_present = 'graphs/track_graph' in f

        if not track_graph_present:

            print 'create track graph'
            
            cells = np.array(f['volumes/labels/cells'])
            lineages = np.array(f['volumes/labels/lineages'])
            
            cells[mask==1] = 0
            lineages[mask==1] = 0

            cells = remove_small_labels(cells, ignore_px)
            lineages[cells==0] = 0

            if 'dilate3d' in process:
                cells, lineages = apply_3d_greyscale_dilation(cells, lineages)

            if 'watershed3d' in process:
                cells, lineages = apply_3d_watershed(raw, cells, lineages, mask)
            
            cells[mask==1] = 0
            lineages[mask==1] = 0

            unique_cells = relabel_cells(cells, ignore_px)
            lineages[unique_cells==0] = 0
            
            if 'dilate' in process:
                unique_cells, lineages = apply_grey_dilation(unique_cells, lineages, 1)
            
            if 'watershed' in process:
                unique_cells, lineages = apply_watershed(raw, unique_cells, lineages, mask)

            unique_cells[mask==1] = 0
            lineages[mask==1] = 0

            f.create_dataset(
                    'volumes/labels/unique_ids',
                    data = unique_cells,
                    compression = 'gzip')
            
            # create track graph
            edges, edge_weights = create_candidate_graph(unique_cells, cells, lineages)
            tracks, track_graph = greedy_track(unique_cells, edges, edge_weights, 0)
            
            track_graph_data = np.array([
                [
                    t.label,
                    t.start,
                    t.end,
                    t.parent.label if t.parent is not None else 0
                ]
                for t in track_graph
            ], dtype=np.uint64)
            
            f.create_dataset(
                'volumes/labels/tracks',
                data=tracks,
                compression="gzip")
            f.create_dataset(
                'graphs/track_graph',
                data=track_graph_data,
                compression="gzip")


        with h5py.File(res_file, 'r') as f:
            res_tracks = np.array(f['volumes/labels/tracks'])
            res_track_graph = np.array(f['graphs/track_graph'])
            res_tracks[mask==1] = 0
     
        output_errors = True
        if output_errors:
            report, vertex_errors, edge_errors = get_tra(res_tracks, res_track_graph, 
                    gt_tracks, gt_track_graph, output_errors)
            f = h5py.File(res_file, 'r+')
            f.create_dataset(
                    'volumes/labels/vertex_errors',
                    data=vertex_errors,
                    compression="gzip")
            f.create_dataset(
                    'volumes/labels/edge_errors',
                    data=edge_errors,
                    compression="gzip")
        else:
            report = get_tra(res_tracks, res_track_graph, gt_tracks, gt_track_graph)
        
        #report = evaluate_segtra(res_tracks, res_track_graph, gt_tracks, gt_track_graph)

    if output_file is not None:
        report_file = output_file + '.json'
    else:
        report_file = res_file[:-4] + 'json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("Saved report %s in %s"%(report, report_file))

if __name__ == "__main__":

    args = parse_arguments()
    print 'evaluating with following parameters:'
    for a in args.__dict__:
        print(str(a) + ': ' + str(args.__dict__[a]))

    evaluate_files(args.res_file, args.gt_file, args.output_file, args.output_errors,
            args.process, args.ignore_px, args.recreate_track)
