import numpy as np
import tempfile
import os
import shutil
from PIL import Image
from subprocess import check_output, CalledProcessError

class TooManyTracksError(RuntimeError):
    pass

def write_track_file(tracks, filename):

    with open(filename, 'w') as f:
        for track in tracks:
            f.write( '%d %d %d %d\n'%tuple(track))

def relabel(tracks, track_graph):

    labels = list(np.unique(tracks))
    if 0 in labels:
        labels.remove(0)

    if len(labels) >= 2**16:
        print("Track graph contains %d distinct labels, can not be expressed "
              "in int16. Skipping evaluation."%len(labels))
        raise TooManyTracksError()

    old_values = np.array(labels)
    new_values = np.arange(1, len(labels) + 1, dtype=np.uint16)

    values_map = np.arange(int(tracks.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    tracks = values_map[tracks]
    for track in track_graph:
        track[0] = values_map[track[0]] # label
        if track[3] != 0:
            track[3] = values_map[track[3]] # parent label

    return tracks, track_graph

def evaluate_segtra(res_tracks, res_track_graph, gt_tracks, gt_track_graph):

    try:

        # holy cow, they need 16-bit encodings!
        if res_tracks.max() >= 2**16:
            print("Converting res to int16... m(")
            res_tracks, res_track_graph = relabel(res_tracks, res_track_graph)
        if gt_tracks.max() >= 2**16:
            print("Converting gt to int16... m(")
            gt_tracks, gt_track_graph = relabel(gt_tracks, gt_track_graph)

    except TooManyTracksError:

        print('Error in relabeling!')
        return {
            'seg_score': np.nan,
            'tra_score': np.nan,
            'error': 'too many tracks for evaluation'
        }

    res_tracks = res_tracks.astype(np.uint16)
    gt_tracks = gt_tracks.astype(np.uint16)

    print(res_tracks.dtype, gt_tracks.dtype)

    # create a temp dir
    dataset_dir = tempfile.mkdtemp()
	
    print("Using temp dir %s"%dataset_dir)

    try:

        res_dir = os.path.join(dataset_dir, '01_RES')
        gt_dir = os.path.join(dataset_dir, '01_GT', 'SEG')
        gt_track_dir = os.path.join(dataset_dir, '01_GT', 'TRA')

        os.makedirs(res_dir)
        os.makedirs(gt_dir)
        os.makedirs(gt_track_dir)

        # store seg and gt as stack of tif files...
        assert res_tracks.shape[0] == gt_tracks.shape[0]

        # FORMAT:
        #
        # GT segmentation:
        #   * background 0
        #   * objects with IDs >=1, 16bit...
        #   -> this is what we already have
        #
        # RES segmentation:
        #   * background 0
        #   * objects with unique IDs >=1 in 2D, change between frames
        #     (hope this is not necessary, we will run out of IDs due to 16-bit
        #     encoding...)

        print("Preparing files for evaluation binaries...")
        for z in range(res_tracks.shape[0]):

            res_outfile = os.path.join(res_dir, 'mask%03d.tif'%z)
            gt_outfile = os.path.join(gt_dir, 'man_seg%03d.tif'%z)
            gt_track_outfile = os.path.join(gt_track_dir, 'man_track%03d.tif'%z)

            res_im = Image.fromarray(res_tracks[z].astype('uint16'))
            gt_im = Image.fromarray(gt_tracks[z].astype('uint16'))
            res_im.save(res_outfile)
            gt_im.save(gt_outfile)
            gt_im.save(gt_track_outfile)

        print("Computing SEG score...")
        try:

            seg_output = check_output([
                './segtra_measure/Linux/SEGMeasure',
                dataset_dir,
                '01'
            ])

        except CalledProcessError as exc:

            print("Calling SEGMeasure failed: ", exc.returncode, exc.output)
            seg_score = 0

        else:

            seg_score = float(seg_output.split()[2])

        print("SEG score: %f"%seg_score)

        write_track_file(res_track_graph, os.path.join(res_dir, 'res_track.txt'))
        write_track_file(gt_track_graph, os.path.join(gt_track_dir, 'man_track.txt'))

        print("Computing TRA score...")
        try:

            tra_output = check_output([
                './segtra_measure/Linux/TRAMeasure',
                dataset_dir,
                '01'
            ])

        except CalledProcessError as exc:

            print("Calling TRAMeasure failed: ", exc.returncode, exc.output)
            tra_score = 0

        else:

            tra_score = float(tra_output.split()[2])

        print("TRA score: %f"%tra_score)

    finally:

        shutil.rmtree(dataset_dir)

    return {
        'seg_score': seg_score,
        'tra_score': tra_score,
    }
