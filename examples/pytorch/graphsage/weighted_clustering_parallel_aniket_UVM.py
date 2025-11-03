import numpy as np
import pandas as pd
import sys
import dgl
import time
import torch as th
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
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# ------------------------------------- Custom CuPy Kernel ----------------#
similarity_kernel = cp.RawKernel(r'''
extern "C" __global__
void similarity_kernel(
    const float* node_features, const int* row_ptr, const int* col_idx,
    const int* edges_src, const int* edges_dst, float* cosine_sim, float* jaccard_sim,
    int num_edges, int num_nodes, int feature_dim
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= num_edges) return;

    int src = edges_src[edge_idx];
    int dst = edges_dst[edge_idx];

    // Cosine similarity
    float dot = 0.0f, norm_src = 0.0f, norm_dst = 0.0f;
    for (int i = 0; i < feature_dim; i++) {
        float src_val = node_features[src * feature_dim + i];
        float dst_val = node_features[dst * feature_dim + i];
        dot += src_val * dst_val;
        norm_src += src_val * src_val;
        norm_dst += dst_val * dst_val;
    }
    norm_src = sqrtf(norm_src);
    norm_dst = sqrtf(norm_dst);
    cosine_sim[edge_idx] = (norm_src > 0.0f && norm_dst > 0.0f) ? max(dot / (norm_src * norm_dst), 0.0f) : 0.0f;

    // Jaccard similarity using sorted neighbor comparison
    int src_start = row_ptr[src];
    int src_end = row_ptr[src + 1];
    int dst_start = row_ptr[dst];
    int dst_end = row_ptr[dst + 1];

    int intersection = 0;
    int i = src_start, j = dst_start;
    while (i < src_end && j < dst_end) {
        int src_neighbor = col_idx[i];
        int dst_neighbor = col_idx[j];
        if (src_neighbor == dst_neighbor) {
            intersection++;
            i++;
            j++;
        } else if (src_neighbor < dst_neighbor) {
            i++;
        } else {
            j++;
        }
    }
    int union_count = (src_end - src_start) + (dst_end - dst_start) - intersection;
    jaccard_sim[edge_idx] = (union_count > 0) ? (float)intersection / union_count : 0.0f;
    //Delete variables
    //del edge_idx,src,dst,dot,norm_src,norm_dst,src_end,src_start,dst_end,dst_start,intersection,i,j,union_count;
    //Run garbage collection
    //gc.collect();

    //Free memory from CuPy's memory pool
    //cp._default_memory_pool.free_all_blocks();
}
''', 'similarity_kernel')

# ------------------------------------- Graph CONSTRUCTION USING data ----------------#
totalTime = 0
start = time.time()
start1 = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
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
        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    G = data[0]
    node_feature = G.ndata["feat"]
    node_label = G.ndata["label"]
    # Check GPU memory
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    print(f"Free GPU memory: {free_mem // (1024**2)} MB")
    print(f"Total GPU memory: {total_mem // (1024**2)} MB")
    # Step 1: Get shape
    #node_feature = G.ndata["feat"]
    print(node_feature.shape, node_feature.dtype)
    print("Total bytes:", node_feature.numel() * node_feature.element_size())


    node_np = node_feature.cpu().numpy()
    shape = node_np.shape
    dtype = node_np.dtype
    nbytes = node_np.nbytes
    ptr = cp.cuda.runtime.mallocManaged(nbytes, cp.cuda.runtime.cudaMemAttachGlobal)
    memptr = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, nbytes, owner=None), 0)
    unified_feat_cp = cp.ndarray(shape=shape, dtype=dtype, memptr=memptr)
    unified_feat_cp.set(node_np)
    print("Unified-memory array successfully created and filled.")
    unified_feat_torch = torch.as_tensor(unified_feat_cp)
    print(unified_feat_torch.device)  # cuda:0
    # try:
    #     node_feature = node_feature.to('cuda')
    # except RuntimeError as e:
    #     print(f"CUDA error moving features to GPU: {e}")
    #     print("Clearing GPU memory and retrying...")
    #     th.cuda.empty_cache()
    #     node_feature = node_feature.to('cuda')
    f = G.ndata["feat"].shape[1]
    print("f={}".format(f))

    # Finding nodes with feature vectors of all zeros
    # zeros_tensor = torch.zeros(f, device='cuda')
    # equality_mask = torch.all(node_feature == zeros_tensor, dim=1)
    # index = torch.nonzero(equality_mask, as_tuple=True)[0]
    # print("Number of feature vector of zeros is {}".format(len(index)))

    size = G.num_edges()
    vertices = G.num_nodes()
    print(f"Number of nodes and edges are {vertices},{size}")

    mem_usage = (psutil.Process().memory_info().rss) / (1024 * 1024 * 1024)
    print(f"Current memory usage: {mem_usage} GB")
    end = time.time()
    totalTime = totalTime + (end - start)
    print("Data Loading Successful!!!! \tTime Taken of Loading is :", round((end - start), 4), "Seconds")
    mem_usage = (psutil.Process().memory_info().rss) / (1024 * 1024 * 1024)
    print(f"Current memory usage: {mem_usage} GB")

    # ---------------------- DGL PREPROCESS -----------------------------------#
    Nodes = G.num_nodes()
    Edges = G.num_edges()
    row_ptr = np.array(G.adj_tensors('csr')[0])
    col_idx = np.array(G.adj_tensors('csr')[1])
    row_ptr_s = len(row_ptr)
    col_idx_s = len(col_idx)
    print(row_ptr_s)
    print(col_idx_s)
    end = time.time()
    totalTime = totalTime + (end - start)
    print("Graph Construction Successful!!!! \tTime Taken :", round((end - start), 4), "Seconds")

    # Convert to CuPy arrays
    try:
        row_ptr = cp.asarray(row_ptr, dtype=cp.int32)
        col_idx = cp.asarray(col_idx, dtype=cp.int32)
        #col_idx = cp.sort(col_idx)  # Ensure sorted for Jaccard
        #node_feature_cupy = cp.asarray(node_feature, dtype=cp.float32)
        #node_feature_cupy = cp.asarray(unified_feat_torch, dtype=cp.float32)
    except cp.cuda.memory.OutOfMemoryError:
        print("CuPy out of memory. Clearing GPU memory and retrying...")
        cp._default_memory_pool.free_all_blocks()
        row_ptr = cp.asarray(row_ptr, dtype=cp.int32)
        col_idx = cp.asarray(col_idx, dtype=cp.int32)
        #col_idx = cp.sort(col_idx)
        #node_feature_cupy = cp.asarray(node_feature, dtype=cp.float32)
        #node_feature_cupy = cp.asarray(unified_feat_torch, dtype=cp.float32)

    print("row_ptr size:", len(row_ptr))
    print("col_idx size:", len(col_idx))

    # Initialize CuPy arrays for similarities
    try:
        cosine_similarities = cp.zeros(size, dtype=cp.float32)
        jaccard_similarity = cp.zeros(size, dtype=cp.float32)
    except cp.cuda.memory.OutOfMemoryError:
        print("CuPy out of memory for similarity arrays. Exiting...")
        sys.exit(1)

    # Precompute edge list for kernel
    start_time = time.time()
    edges = []
    for i in range(len(row_ptr) - 1):
        start = row_ptr[i].item()
        end = row_ptr[i + 1].item()
        for j in range(start, end):
            edges.append((i, col_idx[j].item()))
    edges_src = cp.array([e[0] for e in edges], dtype=cp.int32)
    edges_dst = cp.array([e[1] for e in edges], dtype=cp.int32)
    end_time = time.time()
    print("CSR to COO time ", end_time - start_time)

    # Launch kernel
    block_size = 128  # Reduced for stability
    grid_size = (size + block_size - 1) // block_size
    start_time = time.time()
    similarity_kernel(
            (grid_size,), (block_size,),
            (
                unified_feat_cp, row_ptr, col_idx, edges_src, edges_dst,
                cosine_similarities, jaccard_similarity, size, vertices, f
            )
        )
    end_time = time.time()
    # try:
    #     similarity_kernel(
    #         (grid_size,), (block_size,),
    #         (
    #             unified_feat_cp, row_ptr, col_idx, edges_src, edges_dst,
    #             cosine_similarities, jaccard_similarity, size, vertices, f
    #         )
    #     )
    #     #cp.cuda.Stream.null.synchronize()
    # except cp.cuda.memory.OutOfMemoryError:
    #     print("CuPy out of memory during kernel launch. Exiting...")
    #     sys.exit(1)

    # Validate similarity outputs
    print("kernel time ", end_time - start_time)
    cosine_stats = cp.array([cp.min(cosine_similarities), cp.max(cosine_similarities), cp.mean(cosine_similarities)])
    jaccard_stats = cp.array([cp.min(jaccard_similarity), cp.max(jaccard_similarity), cp.mean(jaccard_similarity)])
    print("Cosine similarity stats (min, max, mean):", cp.asnumpy(cosine_stats))
    print("Jaccard similarity stats (min, max, mean):", cp.asnumpy(jaccard_stats))
    start_time = time.time()
    # Combine similarities and compute weights
    weight_vector1 = cosine_similarities + jaccard_similarity
    weight_vector = weight_vector1 * 50
    weight_vector = cp.rint(weight_vector).astype(cp.int64)

    #printing the weighted vector
    print("weight vector", weight_vector)
    end_time = time.time()
    print("Weight calculation time ", end_time - start_time)
    # Save weight_vector for validation
    output_dir = f'/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_cupy_{args.num_clusters}'
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/{args.dataset}_weight_vector.npy', cp.asnumpy(weight_vector))

    print("weight_vector sample (first 10):", cp.asnumpy(weight_vector)[:10])
    #end_time = time.time()
    print("Weight calculation times: ", end_time - start_time)

    # Convert weight_vector back to PyTorch for DGL compatibility
    weight_vector_torch = th.tensor(cp.asnumpy(weight_vector), dtype=th.int64)
    G.edata['weight'] = weight_vector_torch

    # Convert to lists for PyMetis
    xadj = cp.asnumpy(row_ptr).tolist()
    adjncy = cp.asnumpy(col_idx).tolist()
    adjwgt = cp.asnumpy(weight_vector).tolist()

    nopart = args.num_clusters
    print("Start Partitioning Weight_graph.....")
    start = time.time()
    try:
        node_parts_weight = dgl.metis_partition_assignment(G, nopart)
    except Exception as e:
        print(f"METIS partitioning failed: {e}")
        sys.exit(1)
    end = time.time()
    totalTime = totalTime + (end - start)
    print("Partition is Done !!!!!\t Time of Partition is :", round((end - start), 4), "Seconds")
    mem_usage = (psutil.Process().memory_info().rss) / (1024 * 1024 * 1024)
    print(f"Current memory usage: {mem_usage} GB")

    # Free CuPy memory
    #del row_ptr, col_idx, node_feature_cupy, cosine_similarities, jaccard_similarity, weight_vector, weight_vector1, edges_src, edges_dst
    del row_ptr, col_idx, cosine_similarities, jaccard_similarity, weight_vector, weight_vector1, edges_src, edges_dst
    cp._default_memory_pool.free_all_blocks()

    end1 = time.time()
    print("Preprocess Successful!!!! \tTime Taken of Preprocess is :", round((end1 - start1), 4), "Seconds")

    # Cluster processing
    start_time = time.time()
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
    start_time = time.time()
    #node_feature = node_feature_cupy.cpu()
    for j, row in enumerate(clusters):
        #y = node_feature[row]
        y = unified_feat_torch[row]
        summ = th.zeros(f)
        for p in y:
            summ += p
        r = summ / len(y)
        representative = th.cat((representative, r.view(1, -1)), dim=0)
    end_time = time.time()
    print("representative formation time ", end_time - start_time)
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
