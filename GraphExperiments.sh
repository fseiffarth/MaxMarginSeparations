#!/bin/bash
python main_run_graph.py -is_tree true -runs 1000 -graph_sizes 500 1000 2000 5000 10000 20000 -class_numbers 2 3 5 10 -train_sizes 1 2 3 5 10 -database_name Test 
