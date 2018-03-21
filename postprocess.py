import sys
import numpy as np
import h5py
from scipy import ndimage
from greedy_track import greedy_track
from skimage.morphology import watershed
#from skimage.segmentation import random_walker


def replace(array, old_values, new_values):

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def find_centers(ids):

    all_ids = np.unique(ids)
    coms = ndimage.center_of_mass(np.ones_like(ids), ids, all_ids)

    return { i: l for i, l in zip(all_ids, coms) if i != 0 }


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def remove_small_labels(cells, num_pixel):
    unique, counts = np.unique(cells, return_counts=True)
    small_labels = unique[counts <= num_pixel]
    print "remove small labels: ", len(small_labels)

    cells = replace(cells, small_labels, np.zeros((len(small_labels)), dtype=np.uint64))
    return cells.astype(np.uint64)


def remove_unconnected_pixels(cells, num_pixel):

    print 'remove unconnected pixels'
    labels = list(np.unique(cells))
    if 0 in labels:
        labels.remove(0)

    for label in labels:
        
        idx = cells == label
        bounds = ndimage.find_objects(idx)[0]

        comp, num_comp = ndimage.label(idx[bounds], structure=np.ones((3,3,3)))
        unique, counts = np.unique(comp[comp > 0], return_counts=True)
        for i in unique[counts <= num_pixel]:
            cells[bounds][comp == i] = 0
    
    return cells


def relabel_cells(cells, num_pixel):

    unique_cells = np.zeros_like(cells)
    current_label = 1
    
    print 'relabel cells with unique id in 2d'
    print num_pixel
    for z in range(cells.shape[0]):
                
        labels = list(np.unique(cells[z]))
        if 0 in labels:
            labels.remove(0)
        for label in labels:
            comp, num_comp = ndimage.label(cells[z] == label, structure=np.ones((3,3)))
            for i in range(num_comp):
                idx = comp == i + 1
                if np.sum(idx) > num_pixel:
                    unique_cells[z][idx] = current_label
                    current_label += 1

    return unique_cells


def apply_grey_dilation(unique_cells, lineages, iteration=1):

    print 'apply grey dilation'
    
    for z in range(unique_cells.shape[0]):
        overlap = np.array([unique_cells[z].flatten(), lineages[z].flatten()])
        labels = np.transpose(np.unique(overlap, axis=1))

        for i in range(iteration):
            unique_cells[z] = ndimage.grey_dilation(unique_cells[z], size=(3,3))
        
        for cell_label, lineage_label in labels:
            if cell_label != 0:
                lineages[z][unique_cells[z]==cell_label] = lineage_label
    
    return unique_cells, lineages


def apply_3d_grey_dilation(cells, lineages, iteration=1):

    print 'apply 3d grey dilation'

    overlap = np.array([unique_cells[z].flatten(), lineages[z].flatten()])
    labels = np.transpose(np.unique(overlap, axis=1))

    for i in range(iteration):
        cells = ndimage.grey_dilation(cells, size=(3,3,3))

    for cell_label, lineage_label in labels:
        if cell_label != 0:
            lineages[unique_cells==cell_label] = lineage_label

    return cells, lineages


def apply_watershed(raw, unique_cells, lineages, mask=None):
    
    print 'apply watershed'

    for z in range(unique_cells.shape[0]):
        overlap = np.array([unique_cells[z].flatten(), lineages[z].flatten()])
        labels = np.transpose(np.unique(overlap, axis=1))
        raw[z] = ndimage.gaussian_filter(raw[z], sigma=1)
        
        if mask is not None:
            unique_cells[z] = watershed(raw[z], unique_cells[z], np.ones((3,3)), 
                    mask=np.logical_not(mask[z]))
        else:
            unique_cells[z] = watershed(raw[z], unique_cells[z], np.ones((3,3)))
        
        
        #unique_cells[z] = ndimage.watershed_ift(raw[z].astype(np.uint8), 
        #        unique_cells[z].astype(np.int), structure=np.ones((3,3)))
        #unique_cells[z] = random_walker(raw[z], unique_cells[z])

        for cell_label, lineage_label in labels:
            if cell_label != 0:
                lineages[z][unique_cells[z]==cell_label] = lineage_label
    
    return unique_cells, lineages


def apply_3d_watershed(raw, cells, lineages, mask=None):

    print 'apply 3d watershed'

    overlap = np.array([cells.flatten(), lineages.flatten()])
    labels = np.transpose(np.unique(overlap, axis=1))

    for z in range(raw.shape[0]):
        raw[z] = ndimage.gaussian_filter(raw[z], sigma=1)
    
    if mask is not None:
        cells = watershed(raw, cells, np.ones((3,3,3)), mask=np.logical_not(mask))
    else:
        cells = watershed(raw, cells, np.ones((3,3,3)))

    #cells = ndimage.watershed_ift(raw.astype(np.uint8), cells.astype(np.int), 
    #        structure=np.ones((3,3,3)))
    #cells = random_walker(raw, cells, mode='cg_mg')
    
    for cell_label, lineage_label in labels:
        if cell_label != 0:
            lineages[cells==cell_label] = lineage_label

    return cells, lineages


def create_candidate_graph(unique_cells, cells, lineages):
   
    print 'create candidate graph'
    edges = []
    edge_weights = []
    
    for z in range(cells.shape[0]-1):
        
        print 'frame: ', z 
        frame_edges = []
        frame_weights = []

        # find edges between frames
        current_labels, current_label_idx = np.unique(unique_cells[z], return_index=True)
        successor_labels, successor_label_idx = np.unique(unique_cells[z+1], return_index=True)
        
        overlay = np.array([unique_cells[z].flatten(), lineages[z].flatten()])
        labels  = np.transpose(np.unique(overlay, axis=1))
        
        overlay = np.array([lineages[z+1].flatten(), unique_cells[z+1].flatten()])
        next_labels = np.unique(overlay, axis=1)

        overlay = np.array([unique_cells[z].flatten(), unique_cells[z+1].flatten()])
        iou, counts = np.unique(overlay, return_counts=True, axis=1)
        
        for i in range(len(labels)):
            label = labels[i]
            if label[0] == 0:
                continue

            candidates = list(next_labels[1][next_labels[0]==label[1]])
            if 0 in candidates:
                candidates.remove(0)
            
            for candidate in candidates:
                frame_edges.append((label[0], candidate))
                weight = np.sum(counts[np.logical_and(iou[0]==label[0], iou[1]==candidate)]) / float(np.sum(counts[np.logical_or(iou[0]==label[0], iou[1]==candidate)]))
                
                # add one if cells are within the same body
                current_body_label = cells[z].flatten()[current_label_idx[current_labels==label[0]]]
                next_body_label = cells[z+1].flatten()[successor_label_idx[successor_labels==candidate]]
                
                if current_body_label == next_body_label:
                    weight += 1
                
                frame_weights.append(weight)

        edges.append(np.array(frame_edges, dtype=np.uint64))
        edge_weights.append(np.array(frame_weights, dtype=float))


    return edges, edge_weights

    

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Usage: postprocess.py <res_file>"

    res_file = sys.argv[1]


    f = h5py.File(res_file, 'a') #r+
    cells = np.array(f['volumes/labels/cells'])
    lineages = np.array(f['volumes/labels/lineages'])
    
    """
    if 'volumes/labels/unique_ids' in f:
        del f['volumes/labels/unique_ids']
    if 'volumes/labels/cleaned' in f:
        del f['volumes/labels/cleaned']
    if 'volumes/labels/closed' in f:
        del f['volumes/labels/closed']
    """

    if 'graphs/candidate_graph' in f:
        del f['graphs/candidate_graph']
    if 'graphs/candidate_graph_edges' in f:
        del f['graphs/candidate_graph_edges']
    if 'volumes/labels/tracks' in f:
        del f['volumes/labels/tracks']
    if 'graphs/track_graph' in f:
        del f['graphs/track_graph']

    if 'volumes/labels/cleaned' not in f:
        
        cells = remove_unconnected_pixels(cells)
        f.create_dataset(
                'volumes/labels/cleaned',
                data = cells,
                compression = 'gzip')

    if 'volumes/labels/unique_ids' not in f:
       
        unique_cells = relabel_cells(cells)
        f.create_dataset(
                'volumes/labels/unique_ids',
                data = unique_cells,
                compression = 'gzip')
    else:
        unique_cells = np.array(f['volumes/labels/unique_ids'])

    
    edges, edge_weights = create_candidate_graph(unique_cells, cells, lineages)
    #edges_data = np.asarray(edges, dtype=np.uint64)
    #edge_weights_data = np.asarray(edge_weights, dtype=float)
    """
    f.create_dataset(
            'graphs/candidate_graph',
            data=edges_data,
            compression='gzip')
    f.create_dataset(
            'graphs/candidate_graph_edges',
            data=edge_weights_data,
            compression='gzip')
    """

    #create track graph
    
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

