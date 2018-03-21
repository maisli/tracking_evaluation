# tracking_evaluation
Reimplementation of TRA measure
https://www.nature.com/articles/nmeth.4473

based  on Acyclic Oriented Graph Matching (AOGM):
https://www.nature.com/articles/nmeth.4473

graph creation and morality checks copied from:
https://github.com/funkey/flywing/tree/master/04_evaluate

to start evaluation:
python evaluate.py -r <result_file> -g <gt_file>

further parameter: -o <output_file> -p <[dilate|watershed|dilate3d|watershed3d]> -i <ignore_pixel> --recreate_track

