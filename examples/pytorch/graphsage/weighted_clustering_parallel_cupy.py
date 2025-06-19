import numpy as np
import pandas as pd
import sys
import dgl
import time
import torch as th
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
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset,WisconsinDataset,FlickrDataset,RedditDataset,YelpDataset


#-------------------------------------Graph CONSTRUCTION USING data----------------#
totalTime =0
start = time.time()
start1 = time.time()
# file_name, file_extension = os.path.splitext(sys.argv[1])
# print(file_extension)
# suffix_csr = "_output.csr"
# suffix_part = "_reorder.SHEM."
# suffix_part1 = "_reorder.RM."
# file_name = file_name.split("/")
# file_name = file_name[len(file_name)-1]
# out_filename1 = str(file_name) + suffix_csr
# #out_filename2 = str(file_name) + suffix_part + str(sys.argv[2]) + ".csv"
# out_filename3 = str(file_name) + suffix_part1 + str(2) + ".csv"
#print(out_filename2)
if __name__ == "__main__":
    start=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        #help="Dataset name ('cora', 'citeseer', 'pubmed', 'wisconsin','flickr', 'reddit', 'yelp', 'ogbn-products','ogbn-arxiv', 'papers').",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=10,
        help="Number of cluster",
    )

    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
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
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    G = data[0]
    features_tensor = G.ndata["feat"]
    # Convert to CuPy
    features = cp.asarray(features_tensor)  # Works directly
    node_label = G.ndata["label"]
    #node_feature = node_feature.to('cuda')
    #print("node features:",node_feature)
    #print("node labels:",node_label)
    # f=G.ndata["feat"].shape[1]
    # print("f={}".format(f))
    # #finding nodes with feature vectors of all zeros.
    # zeros_tensor=torch.zeros(f, device='cuda')
    #
    # equality_mask = torch.all(features == zeros_tensor, dim=1)
    #
    # # Find the index where all elements are zero
    # index = torch.nonzero(equality_mask, as_tuple=True)[0]
    # print("Number of feature vector of zeros is {}".format(len(index)))
    size = G.num_edges()
    vertices = G.num_nodes()
    print(f"Number of nodes and edges are {vertices},{size}")

    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} bytes")
    end = time.time()
    totalTime = totalTime + (end-start)

    print("Data Loading Successfull!!!! \tTime Taken of Loading is :",round((end-start),4), "Seconds")
    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} GB")

    #----------------------DGL PREPROCESS-----------------------------------#

    Nodes = G.num_nodes()
    Edges = G.num_edges()
    indptr=cp.asarray(G.adj_tensors('csr')[0])
    indices=cp.asarray(G.adj_tensors('csr')[1])
    # row_ptr_s=len(row_ptr)
    # col_idx_s=len(col_idx)
    # print(row_ptr_s)
    # print(col_idx_s)
    end = time.time()
    totalTime = totalTime + (end-start)
    print("Graph Construction Successfull!!!! \tTime Taken :",round((end-start),4), "Seconds")
    num_nodes = features.shape[0]
    feature_dim = features.shape[1]

    # Output edge weights
    edge_weights = cp.zeros(indices.shape, dtype=cp.float32)

    # CUDA kernel
    kernel_code = r'''
    extern "C" __global__
    void edge_weights_kernel(
        const float* __restrict__ features,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        float* edge_weights,
        int num_nodes,
        int feature_dim
    ) {
        int src = blockIdx.x;
        int edge_start = indptr[src];
        int edge_end = indptr[src + 1];

        for (int e = edge_start + threadIdx.x; e < edge_end; e += blockDim.x) {
            int dst = indices[e];

            // Cosine similarity
            float dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (int i = 0; i < feature_dim; ++i) {
                float a = features[src * feature_dim + i];
                float b = features[dst * feature_dim + i];
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            float cosine = (norm_a > 0 && norm_b > 0) ? dot / (sqrtf(norm_a) * sqrtf(norm_b)) : 0.0;

            // Jaccard similarity
            int count_intersection = 0;
            int count_union = 0;
            int i = indptr[src], j = indptr[dst];

            while (i < indptr[src + 1] && j < indptr[dst + 1]) {
                int a = indices[i];
                int b = indices[j];
                if (a == b) {
                    count_intersection++;
                    count_union++;
                    i++; j++;
                } else if (a < b) {
                    count_union++;
                    i++;
                } else {
                    count_union++;
                    j++;
                }
            }
            count_union += (indptr[src + 1] - i) + (indptr[dst + 1] - j);

            float jaccard = count_union > 0 ? (float)count_intersection / count_union : 0.0;

            edge_weights[e] = (cosine + jaccard)*50;
        }
    }
    '''

    kernel = cp.RawKernel(kernel_code, 'edge_weights_kernel')
    threads_per_block = 128
    blocks = num_nodes
    start_time = time.time()

    kernel((blocks,), (threads_per_block,),
           (features.ravel(), indptr, indices, edge_weights, num_nodes, feature_dim))
    # weight_vector = weight_vector1 * 50
    # weight_vector = torch.round(weight_vector)
    # weight_vector = weight_vector.to(torch.int64)
    print("weight_vector",edge_weights)
    #print("Length of weight_vector", len(weight_vector))
  
    #cosine_similarities = torch.where(cosine_similarities < 0, torch.tensor(0.0), cosine_similarities)
    
    end_time = time.time()
    #print(weights)
    #print("Length of weight: ",len(weights))
    #print("Weight tensor: ",weights)
    print("Weight calculation times: ",end_time - start_time)
    '''
    cosine_similarities = cosine_similarities.tolist()
    file_path4 = 'cluster_id.txt'
    with open(file_path4, "w") as file:
        for value in cosine_similarities:
        file.write(f"{value}\n")
    '''    

    # Convert edge list to a tensor for batch processing
    #edges = torch.tensor(edges)
    #print("Edges tensor has been created")

    # Extract features for source and destination nodes
    #src_features = G.ndata['feat'][edges[:, 0]]
    #dst_features = G.ndata['feat'][edges[: ,1]]
    # src_features = node_features[edges[:, 0]]
    # dst_features = node_features[edges[:, 1]]
    #print(type(src_features))
    #print(src_features.shape)
    # Number of elements in the tensor
    #num_elements = src_features.numel()

    # Size of each element in bytes
    #element_size = src_features.element_size()

    # Total memory in bytes
    #total_memory_bytes = num_elements * element_size

    # Convert to gigabytes
    #total_memory_gb = total_memory_bytes / (1024 ** 3)  # Dividing by 2^30 for GB
    #print(total_memory_gb)

    
    # Compute cosine similarity for all edges
    #cosine_similarities = F.cosine_similarity(src_features, dst_features)
    #print("cosine_similarities: ",cosine_similarities)
    #print("size cosine_similarities: ", len(cosine_similarities))
    # Assign the edge weights to the 'weight' attribute of the graph
    #weight_vector = weight_vector.cpu()
    G.edata['weight'] = edge_weights
    #xadj = row_ptr.tolist()  # The xadj array in PyMetis (cumulative degree list)
    #adjncy = col_idx.tolist()  # The adjacency list in PyMetis
    #adjwgt = weight_arr.tolist()# The edge weights in PyMetis
    #adjwgt = weight_vector.tolist() 
    # nopart = int(sys.argv[2])
    #nopart = 20
    nopart = args.num_clusters
    torch.set_printoptions(threshold=torch.inf)
    print("Start Partitioning Weight_graph.....")
    start = time.time()
    node_parts_weight = dgl.metis_partition_assignment(G,nopart)
    #n_cuts, membership = pymetis.part_graph(nopart, xadj=iptr, adjncy=indx, eweights=weights)
    # cut, membership = pymetis.part_graph(nopart, adjacency=None, xadj=xadj, adjncy=adjncy, vweights=None, eweights=adjwgt)
    # cut, membership = pymetis.part_graph(5, xadj=xadj, adjncy=adjncy, eweights=adjwgt)
    #RG = dgl.reorder_graph(G, node_permute_algo='metis', edge_permute_algo='dst', permute_config={'k':nopart})
    end = time.time()
    totalTime = totalTime + (end-start)
    print("Partition is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")
    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} bytes")
    #print("node_parts_weight: ",node_parts_weight)
    # print("cuts: ", cut)
    # print("membership: ", membership)


    # node_parts_weight = np.sort(node_parts_weight)
    # node_parts = np.sort(node_parts)

    # row_ptr=np.array(RG.adj_sparse('csr')[0])
    # col_idx=np.array(RG.adj_sparse('csr')[1])
    # row_ptr_s=len(row_ptr)
    # col_idx_s=len(col_idx)


    #del g_row_ptr
    #del g_col_idx
    #del row_ptr
    #del col_idx
    #del g_weight_arr
    #del g_sum
    #del weight_arr
    #del weight_vector
    #del weight_vector1
    #del sum
    #cp.cuda.runtime.free(intptr_t temp_arr)
    cp._default_memory_pool.free_all_blocks()
    end1 = time.time()
    print("Preprocess Successfull!!!! \tTime Taken of Prepr weight_vector1 = torch.add(cosine_similarities,jaccard_similarity)ocess is :",round((end1-start1),4), "Seconds")
    # Get unique values and the inverse indices (where each value was found)
    unique_values, inverse_indices = node_parts_weight.unique(return_inverse=True)

    # Determine the number of unique values
    num_unique_values = unique_values.size(0)

    # Create a list to hold the indices for each unique value
    clusters = []

    # Iterate through unique values and collect indices
    for unique_value in unique_values:
        # Get the indices of the current unique value
        indices = (inverse_indices == unique_value).nonzero(as_tuple=True)[0]
        clusters.append(indices)
    #clusters = torch.tensor(clusters)
    #clusters = clusters.to('cuda')

    # Find the maximum number of indices to pad the result
    #max_length = max(len(indices) for indices in indices_list)

    # Create a 2D tensor with padding
    #result_tensor = torch.full((num_unique_values, max_length), fill_value=-1, dtype=torch.long)

    # Populate the 2D tensor with the indices
    #for i, indices in enumerate(indices_list):
        #result_tensor[i, :len(indices)] = indices

    #print(type(clusters))
    #print("clusters")
    #print(clusters)
    #representative = []
    representative = torch.empty(0, f)
    #representative = representative.to('cuda')
    #node_feature = node_feature.cpu()
    for j, row in enumerate(clusters):
        #print(j)
        y=features[row]
        summ=torch.zeros(f)
        for p in y:
            summ+=p
        r=summ/len(y)
        representative = torch.cat((representative, r.view(1, -1)), dim=0)
    representative = representative.tolist()
        #del y
        #del summ
        #del r
    #representative = representative.tolist();
    #PRINTING REPRESENTIVE OF EACH CLUSTER
    
    # for j, row in enumerate(representative):
    #     print("representative of {} is".format(j))
    #     print(row)
    #     print("Length of representative",len(row))
    # file_path2 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_20/papers/representative.npy'
    file_path2 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_cupy_' + str(args.num_clusters) +'/' + args.dataset + '_representative.npy'
    # Convert the list to a NumPy array
    np_representative = np.array(representative)

    # Save the NumPy array to a .npy file
    np.save(file_path2, np_representative)
    node_parts_weight = node_parts_weight.tolist()
    # file_path1 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_20/papers/cluster_id.txt'
    file_path1 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_cupy_' + str(args.num_clusters) + '/' + args.dataset + '_cluster_id.txt'
    with open(file_path1, "w") as file:
        for value in node_parts_weight:
            file.write(f"{value}\n")

