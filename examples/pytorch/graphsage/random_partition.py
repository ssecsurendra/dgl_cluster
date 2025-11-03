import numpy as np
import pandas as pd
import sys
import dgl
import time
import torch as th
import pymetis
#import gc
from scipy.io import mmread
import os
from tqdm import tqdm
import cupy as cp
import psutil
import argparse
os.environ["DGLBACKEND"] = "pytorch"
import torch.nn.functional as F
import torch
import math
import copy
import random
import pymetis
import dgl.data
# from . import backend as F, utils
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# ------------------------------------- Graph CONSTRUCTION USING data ----------------#
totalTime = 0
start = time.time()
start1 = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=20,
        help="Number of clusters",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type (float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # Load and preprocess dataset
    try:
        if args.dataset == "cora":
            data = CoraGraphDataset()
        elif args.dataset == "citeseer":
            data = CiteseerGraphDataset()
        elif args.dataset == "pubmed":
            data = PubmedGraphDataset()
        elif args.dataset == "wisconsin":
            data = WisconsinDataset()
        elif args.dataset == "flickr":
            data = FlickrDataset()
        elif args.dataset == "reddit":
            data = RedditDataset()
        elif args.dataset == "yelp":
            data = YelpDataset()
        elif args.dataset == "ogbn-products":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        elif args.dataset == "ogbn-arxiv":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        elif args.dataset == "ogbn-papers":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M"))
        elif args.dataset == "amazon_products":
            load_path = '/data/Dataset/gnn_dataset/amazon_products.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "cit-net":
            load_path = '/data/Dataset/gnn_dataset/citations_network_graph.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-tiny":
            load_path = './dataset/igb_tiny.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-medium":
            load_path = './dataset/igb_medium.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "wiki":
            load_path = './dataset/wikidata5M/wikidata5m_dgl_graph.bin'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-small":
            load_path = './dataset/igb_small.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "amazon_products":
            load_path = './dataset/amazon_products.dgl'
            data, _ = dgl.load_graphs(load_path)       
        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    G = data[0]
        # ---------------------- DGL PREPROCESS -----------------------------------#
    Nodes = G.num_nodes()
    Edges = G.num_edges()
    node_feature1 = G.ndata["feat"]
    f = G.ndata["feat"].shape[1]
    print("# nodes: ",G.num_nodes())
    print("# edges: ",G.num_edges())
    total_start = time.time()
    # ensure undirected
    G = dgl.to_bidirected(G, copy_ndata=True)
    # row_ptr = np.array(G.adj_tensors('csr')[0])
    # col_idx = np.array(G.adj_tensors('csr')[1])
    indptr, indices, _ = G.adj_tensors('csr')
    row_ptr = list(map(int, indptr.numpy()))   # convert to list of ints
    col_idx = list(map(int, indices.numpy()))  # same for adjacency
    print("num_nodes:", G.num_nodes())
    print("len(row_ptr):", len(row_ptr))   # should be num_nodes+1
    print("len(col_idx):", len(col_idx))   # should be 2*#edges for undirected
    print("last row_ptr entry:", row_ptr[-1])  # must equal len(col_idx)
    nopart = args.num_clusters
    print("Start Partitioning Weight_graph.....")
    start = time.time()
    # try:
    #node_parts_weight = dgl.metis_partition_assignment(G, nopart, balance_edges=True)
    # start_bi_dir = time.time()
    # sym_g = dgl.to_bidirected(G, readonly=True)
    # end_bi_dir = time.time()
    # print("Bidirection is Done !!!!!\t Time is :", round((end_bi_dir - start_bi_dir), 4), "Seconds")
    # iptr, indx, eid = sym_g.adj_tensors('csc')   
    # weight_100_np = np.ones(len(col_idx), dtype=np.int32)  # all edges weight 1

    # Step 1: move to CPU and convert to numpy.int32
    # iptr_np = iptr.cpu().numpy().astype(np.int32)
    # indx_np = indx.cpu().numpy().astype(np.int32)
    # weight_100_np = weight_100.cpu().numpy().astype(np.int32)

    # Step 2: convert to list for pymetis
    # iptr = iptr_np.tolist()
    # indx = indx_np.tolist()
    # weight_100 = weight_100_np.tolist()
    #n_cuts, node_parts_weight = pymetis.part_graph(nopart, xadj=iptr, adjncy=indx, eweights=weight_vector_torch)
    # n_cuts, node_parts_weight = pymetis.part_graph(nopart, xadj=row_ptr, adjncy=col_idx, eweights=weight_vector_torch)
    # n_cuts, node_parts_weight = pymetis.part_graph(nopart, xadj=row_ptr, adjncy=col_idx)
    #sequential partitioning
    # l = Nodes // nopart   # calculate the number of repeated values for each number
    #
    # node_parts_weight = np.zeros(Nodes, dtype=int)  # create an array of size n filled with zeros
    #
    # for i in range(nopart):
    #     node_parts_weight[i*l:(i+1)*l] = i  # fill each part of the array with the corresponding number
    #random partitioning    
    node_parts_weight = np.random.randint(0, nopart, size=Nodes)

# except Exception as e:
#     print(f"METIS partitioning failed: {e}")
    # sys.exit(1)
    end = time.time()
    print(type(node_parts_weight))
    #totalTime = totalTime + (end - start)
    print("Partition is Done !!!!!\t Time of Partition is :", round((end - start), 4), "Seconds")
    # Cluster processing
    start_time = time.time()
    node_parts_weight = torch.tensor(node_parts_weight, dtype=torch.int32)
    node_parts_weight = node_parts_weight.clone().detach()  # Fix UserWarning
    unique_values, inverse_indices = node_parts_weight.unique(return_inverse=True)
    num_unique_values = unique_values.size(0)
    clusters = []
    for unique_value in unique_values:
        indices = (inverse_indices == unique_value).nonzero(as_tuple=True)[0]
        clusters.append(indices)
    end_time = time.time()
    print("cluster processing time ",end_time - start_time)
    # Compute representatives
    representative = th.empty(0, f)
    rep_start_time = time.time()
    #node_feature = node_feature_cupy.cpu()
    # for j, row in enumerate(clusters):
    #     #y = node_feature[row]
    #     y = unified_feat_torch[row]
    #     summ = th.zeros(f)
    #     for p in y:
    #         summ += p
    #     r = summ / len(y)
    #     representative = th.cat((representative, r.view(1, -1)), dim=0)
    for node_ids in clusters:
        # print(node_ids)
        # print(node_feature1[node_ids])
        avg_feat = node_feature1[node_ids].mean(dim=0, keepdim=True)  # Shape: (1, F)
        representative = torch.cat([representative, avg_feat], dim=0)    
    rep_end_time = time.time()
    total_end = time.time()
    print("representative formation time ", rep_end_time - rep_start_time)
    print("Total time", total_end-total_start, "Seconds")
    representative = representative.tolist()
    np_representative = np.array(representative)
    file_path2 = f'/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_cupy_{args.num_clusters}/{args.dataset}_representative.npy'
    os.makedirs(os.path.dirname(file_path2), exist_ok=True)
    np.save(file_path2, np_representative)

    node_parts_weight = node_parts_weight.tolist()
    file_path1 = f'/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_cupy_{args.num_clusters}/{args.dataset}_cluster_id.txt'
    os.makedirs(os.path.dirname(file_path1), exist_ok=True)
    with open(file_path1, "w") as file:
        for value in node_parts_weight:
            file.write(f"{value}\n")

         
