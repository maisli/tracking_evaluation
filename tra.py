import numpy as np


def get_pred(track_graph, label, frame):
    
    pred = 0
    pred_frame = None
    pred_link = None

    start, stop, parent = track_graph[label]

    if start == frame:
        pred = parent
        if parent != 0:
            pred_link = 'parent'
            pred_frame = track_graph[parent][1]
    elif start < frame and stop >= frame:
        pred = label
        pred_link = 'track'
        pred_frame = frame - 1

    return pred, pred_frame, pred_link


def replace(array, old_values, new_values):

    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def get_tra_score(res_tracks, res_track_graph, gt_tracks, gt_track_graph, output_errors=False):
    """Create Acyclic Oriented Graph Matching (AOGM)"""
    
    res_dict = {i[0]: i[1:] for i in res_track_graph}
    gt_dict = {i[0]: i[1:] for i in gt_track_graph}
    if len(res_dict) > 0:
        new_idx = max(res_dict, key=int) + 1

    # sum of edit operations
    ns = 0          # vertex splits
    fn = 0          # vertex adding
    fp = 0          # vertex deleting
    ed = 0          # edge deleting
    ea = 0          # edge adding
    ec = 0          # altering adge semantics

    # weights
    w_ns = 5.0      # vertex splitting
    w_fn = 10.0     # vertex adding
    w_fp = 1.0      # vertex deleting
    w_ed = 1.0      # edge deleting
    w_ea = 1.5      # edge adding
    w_ec = 1.0      # altering edge semantics
   
    if output_errors:
        vertex_errors = np.zeros_like(res_tracks)
        edge_errors = np.zeros_like(res_tracks)
    
    for frame in range(gt_tracks.shape[0]):
    
        # get current frame
        res_frame_nodes = res_tracks[frame]
        gt_frame_nodes = gt_tracks[frame]
        
        old_values = []
        new_values = []
        
        # get overlaying res and gt labels
        overlay = np.array([res_frame_nodes.flatten(), gt_frame_nodes.flatten()])
        conn_labels, conn_counts = np.unique(overlay, return_counts=True, axis=1)
        conn_labels = np.transpose(conn_labels)

        gt_labels, gt_counts = np.unique(gt_frame_nodes, return_counts=True)
        res_labels = np.unique(res_frame_nodes)
        
        conn = np.asarray([c > 0.5 * float(gt_counts[gt_labels == v]) 
            for (u,v), c in zip(conn_labels, conn_counts)], dtype=np.bool)
        
        conn_labels = conn_labels[conn]
        res_labels = res_labels[res_labels > 0]
        gt_labels = gt_labels[gt_labels > 0]

        
        # cost to construct gt graph from scratch
        if len(res_labels) == 0:
            
            con_mat = np.reshape(np.zeros((1, len(gt_labels))), (1, len(gt_labels)))
            fn += len(gt_labels)
            
            # count edges
            if frame > 0:
                for gt_label in gt_labels:
                    start, stop, parent = gt_dict[gt_label]
                    if start == frame:
                        if parent != 0:
                            ea += 1
                    elif start < frame:
                        ea += 1
                    else:
                        print "frame outside of track"
                        
        # cost to transform res to gt
        else:
            con_mat = np.zeros((len(res_labels), len(gt_labels)))
            for (u,v) in conn_labels:
                if u > 0 and v > 0:
                    con_mat[np.where(res_labels == u), np.where(gt_labels == v) ] = 1
            # non-split vertices num non-empty cols - num non-empty rows
            ns += np.sum(np.count_nonzero(con_mat, axis=0)) \
                    - np.sum(np.count_nonzero(con_mat, axis=1)>0)
            
            # false negative: empty cols
            fn += np.sum(np.sum(con_mat, axis=0)==0)
            
            # false positive: empty rows
            fp += np.sum(np.sum(con_mat, axis=1)==0)

            if output_errors:
                # ns = 1, fn = 2, fp = 3
                for i in range(len(res_labels)):
                    if np.sum(con_mat[i,:]) > 1:
                        vertex_errors[frame][res_frame_nodes == res_labels[i]] = 1
                    elif np.sum(con_mat[i,:]) == 0:
                        vertex_errors[frame][res_frame_nodes == res_labels[i]] = 3
                for i in range(len(gt_labels)):
                    if np.sum(con_mat[:,i]) == 0:
                        vertex_errors[frame][gt_frame_nodes == gt_labels[i]] = 2

            # delete edges for split or deleted vertices
            for i in range(len(res_labels)):
                
                res_label = res_labels[i]
                start, stop, parent = res_dict[res_label]

                if np.sum(con_mat[i,:]) != 1:
                    
                    # delete track links and incoming parent links
                    if start == frame and stop == frame:
                        res_dict[res_label] = [start, stop, 0]
                    
                    elif start == frame and stop > frame:
                        res_dict[res_label] = [start + 1, stop, 0]
                    
                    elif start < frame and stop == frame:
                        res_dict[res_label] = [start, stop - 1, parent]
                    
                    elif start < frame and stop > frame:
                        res_dict[res_label] = [start, frame-1, parent]
                        res_dict[new_idx] = [frame+1, stop, 0]
                        old_values.append(res_label)
                        new_values.append(new_idx)
                        for child, (child_start, child_stop, child_parent) \
                                in res_dict.iteritems():
                            if child_parent == res_label:
                                res_dict[child] = [child_start, child_stop, new_idx]
                        
                        new_idx += 1
                    
                    else:
                        print "frame outside of track"
                
                    # delete outgoing parent links
                    if stop == frame:
                        for child, (child_start, child_stop, child_parent) \
                                in res_dict.iteritems():
                            if child_parent == res_label:
                                res_dict[child] = [child_start, child_stop, 0]
            
            # evaluate incoming edges
            if frame > 0:
                
                ea_labels = []
                ed_labels = []
                ec_labels = []
                
                # delete edges
                for i in range(len(res_labels)):
                    if np.sum(con_mat[i,:]) == 1:
                        res_label = res_labels[i]
                        res_pred, res_pred_frame, res_pred_link = get_pred(res_dict, res_label, 
                                frame)
                        gt_label = int(conn_labels[conn_labels[:,0]==res_label][:,1])
                        gt_pred, gt_pred_frame, gt_pred_link = get_pred(gt_dict, gt_label, frame)
                        if gt_pred == 0 and res_pred != 0:
                            ed += 1
                            ed_labels.append(res_label)
                        elif gt_pred != 0 and res_pred != 0:
                            if gt_pred_frame != res_pred_frame:
                                ed += 1
                                ed_labels.append(res_label)
                            # alter edge semantics
                            elif res_pred_frame == gt_pred_frame \
                                    and res_pred_link != gt_pred_link:
                                ec += 1
                                ec_labels.append(res_label)

                # add edges and alter edge semantics
                for i in range(len(gt_labels)):

                    gt_label = gt_labels[i]
                    gt_pred, gt_pred_frame, gt_pred_link = get_pred(gt_dict, gt_label, frame)
                    
                    res_pred = 0
                    res_pred_frame = 0
                    res_pred_link = None
                    
                    if gt_label in conn_labels[:,1]:
                        
                        res_label = int(conn_labels[conn_labels[:,1]==gt_label][:,0])
                        if res_label in res_dict:
                            res_pred, res_pred_frame, res_pred_link = get_pred(res_dict, 
                                    res_label, frame)
                    if gt_pred != 0:
                        if res_pred == 0:
                            ea += 1
                            ea_labels.append(gt_label) 
                        elif res_pred_frame != gt_pred_frame:
                            ea += 1
                            ea_labels.append(gt_label)

                if output_errors:
                    # ed = 1, ea = 2, ec = 3
                    for ed_label in ed_labels:
                        edge_errors[frame][res_frame_nodes == ed_label] = 1
                    for ea_label in ea_labels:
                        edge_errors[frame][gt_frame_nodes == ea_label] = 2
                    for ec_label in ec_labels:
                        edge_errors[frame][res_frame_nodes == ec_label] = 3

        
        if len(old_values) > 0:
            old_values = np.array(old_values, dtype=res_tracks.dtype)
            new_values = np.array(new_values, dtype=res_tracks.dtype)
    
            res_tracks[frame+1:,:,:] = replace(res_tracks[frame+1:,:,:], 
                    old_values, new_values)

    
    print ns, fn, fp, ed, ea, ec
    print w_ns * ns, w_fn * fn, w_fp * fp, w_ed * ed, w_ea * ea, w_ec * ec
 
    tra_score = w_ns * ns + w_fn * fn + w_fp * fp + w_ed * ed + w_ea * ea + w_ec * ec
    
    report = {
        'tra_score': tra_score,
        'non_split': ns,
        'false_negative': fn,
        'false_positive': fp,
        'delete_edge': ed,
        'add_edge': ea,
        'alter_edge': ec,
        'weighted_non_split': w_ns * ns,
        'weighted_false_negative': w_fn * fn,
        'weighted_false_positive': w_fp * fp,
        'weighted_delete_edge': w_ed * ed,
        'weighted_add_edge': w_ea * ea,
        'weighted_alter_edge': w_ec * ec
    }


    if output_errors:
        return tra_score, report, vertex_errors, edge_errors
    else:
        return tra_score, report


def get_tra(res_tracks, res_track_graph, gt_tracks, gt_track_graph, output_errors=False):
    
    # tra score to create gt graph from scratch
    empty = np.zeros_like(gt_tracks)
    gt_tra_score, gt_report = get_tra_score(empty, np.array([]), gt_tracks, gt_track_graph)
    
    # tra score to change res graph to gt graph
    if output_errors:
        res_tra_score, res_report, vertex_errors, edge_errors = get_tra_score(res_tracks, 
                res_track_graph, gt_tracks, gt_track_graph, output_errors)
    else:
        res_tra_score, res_report = get_tra_score(res_tracks, res_track_graph,
                gt_tracks, gt_track_graph)

    tra_score = 1 - min(res_tra_score, gt_tra_score) / float(gt_tra_score)
    res_report['total_tra_score'] = tra_score
    
    if output_errors:
        return res_report, vertex_errors, edge_errors
    else:
        return res_report

