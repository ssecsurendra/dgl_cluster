import numpy as np
import cupy as cp
import pandas as pd
import torch
import dgl
import os
import torch.nn.functional as F
os.environ["DGLBACKEND"] = "pytorch"
_computed_array = None
_representative_array = None
_method_array = None
_method_value = 0
_centrality_array = None

def metis_partition(G, dataset_name=None, fan=None):
    global _computed_array
    if _computed_array is None:
        # Perform computation here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
        #print(G)
        #print(type(G))
        #print("partition start")
        
        # dgl.distributed.partition_graph(G, 'test', 4, num_hops=1, part_method='metis', out_path='output/', balance_ntypes=G.ndata['train_mask'], balance_edges=True)
        # ( g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,) = dgl.distributed.load_partition('output/test.json', 0)

        # print(g)
        #_computed_array = dgl.metis_partition_assignment(G, 4, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
        # context = dgl.cuda.get_context(0)
        # context = dgl.cuda.context(0)
        # _computed_array = np.random.rand(10)
        # _computed_array = np.random.randint(10000, 90001, size=10)
        # _computed_array = _computed_array.astype(np.int64)
        # _computed_array = torch.from_numpy(_computed_array)
        # _computed_array = _computed_array.to(device)
        columns = ['Data']
        #file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.txt',names=columns)
        #file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_' + fan + '/' + dataset_name + '_cluster_id.txt',names=columns)
        #file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/aniket/cluster_' + fan + '/' + dataset_name + '_cluster_id.txt',names=columns)
        file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_cupy_' + fan + '/' + dataset_name + '_cluster_id.txt',names=columns)
        #print("This might a take while..")
        #print(file.head())
        Data=file['Data']
        Data=np.array(Data)
        #print(Data.shape)
        # Load PyTorch tensor from .pt file
        #tensor_data = torch.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.pt')

        # Convert PyTorch tensor to NumPy array
        #Data = tensor_data.numpy()
        #Data = np.append(Data, 0)
        _computed_array = dgl.ndarray.array(Data)
        #print("compute array",_computed_array)

        #_computed_array = cp.asarray(_computed_array)
        # _computed_array = _computed_array.to(device)
        # Convert NumPy array to DGL tensor
        # _computed_array = dgl.tensor(_computed_array)
        # device = "cuda" if dgl.cuda.is_available() else "cpu"
        # _computed_array = _computed_array.to(device)
        # _computed_array = _computed_array.tolist
        #print("Array computation done and passed to neighbour.py line 631")
    #else:
        #numpy_array = _computed_array.asnumpy()
        #numpy_array = numpy_array[:-1]
        #numpy_array = np.append(numpy_array, 1)
        #_computed_array = dgl.ndarray.array(numpy_array)
        #print("compute array",_computed_array)
    return _computed_array

def get_method(method=None):
    global _method_array
    global _method_value
    #print("method: ",method)
    if _method_array is None:
       _method_value = method
       _method_array = 1
    return _method_value

def get_representative_array(G, dataset_name=None, fan=None):
    global _representative_array
    if _representative_array is None:
        # Perform computation here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
        #print(G)
        #print(type(G))
        #print("partition start")
        
        # dgl.distributed.partition_graph(G, 'test', 4, num_hops=1, part_method='metis', out_path='output/', balance_ntypes=G.ndata['train_mask'], balance_edges=True)
        # ( g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,) = dgl.distributed.load_partition('output/test.json', 0)

        # print(g)
        #_computed_array = dgl.metis_partition_assignment(G, 4, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
        # context = dgl.cuda.get_context(0)
        # context = dgl.cuda.context(0)
        # _computed_array = np.random.rand(10)
        # _computed_array = np.random.randint(10000, 90001, size=10)
        # _computed_array = _computed_array.astype(np.int64)
        # _computed_array = torch.from_numpy(_computed_array)
        # _computed_array = _computed_array.to(device)
        # following 4 lines read the content of .txt file converts into numpy array.
        #columns = ['Data']
        #file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/representative.txt',names=columns)
        #print(file.head())
        #print("This might a take while..")
        #Data=file['Data']
        #Data=np.array(Data)
        #print(Data.shape)
        #loaded_array = np.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_' + fan + '/' + dataset_name + '_representative.npy')
        #loaded_array = np.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/aniket/cluster_' + fan + '/' + dataset_name + '_representative.npy')
        loaded_array = np.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_cupy_' + fan + '/' + dataset_name + '_representative.npy')
        #loaded_array = np.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/representative.npy')
        #print(loaded_array.shape)
        #print(loaded_array)
        #core.35410print("representataive array type",loaded_array.dtype)

        # Load PyTorch tensor from .pt file
        #tensor_data = torch.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.pt')

        # Convert PyTorch tensor to NumPy array
        #Data = tensor_data.numpy()
        #Data = np.append(Data, 0)
        _representative_array = dgl.ndarray.array(loaded_array)
        #print("compute array",_computed_array)

        #_computed_array = cp.asarray(_computed_array)
        # _computed_array = _computed_array.to(device)
        # Convert NumPy array to DGL tensor
        # _computed_array = dgl.tensor(_computed_array)
        # device = "cuda" if dgl.cuda.is_available() else "cpu"
        # _computed_array = _computed_array.to(device)
        # _computed_array = _computed_array.tolist
        #print("Array computation done and passed to neighbour.py line 631")
    #else:
        #numpy_array = _computed_array.asnumpy()
        #numpy_array = numpy_array[:-1]
        #numpy_array = np.append(numpy_array, 1)
        #_computed_array = dgl.ndarray.array(numpy_array)
        #print("compute array",_computed_array)
    return _representative_array


def get_centrality_array(G, dataset_name=None):
    global _centrality_array
    if _centrality_array is None:
        # Perform computation here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device
        #print(G)
        #print(type(G))
        #print("partition start")
        
        # dgl.distributed.partition_graph(G, 'test', 4, num_hops=1, part_method='metis', out_path='output/', balance_ntypes=G.ndata['train_mask'], balance_edges=True)
        # ( g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,) = dgl.distributed.load_partition('output/test.json', 0)

        # print(g)
        #_computed_array = dgl.metis_partition_assignment(G, 4, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
        # context = dgl.cuda.get_context(0)
        # context = dgl.cuda.context(0)
        # _computed_array = np.random.rand(10)
        # _computed_array = np.random.randint(10000, 90001, size=10)
        # _computed_array = _computed_array.astype(np.int64)
        # _computed_array = torch.from_numpy(_computed_array)
        # _computed_array = _computed_array.to(device)
        # following 4 lines read the content of .txt file converts into numpy array.
        columns = ['Data']

        #loaded_array = np.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_' + fan + '/' + dataset_name + '_representative.npy')
        file = pd.read_csv('/data/Framework/graphsage/POP_FINAL' + '/' + dataset_name + '_degree_centrality.txt', names=columns)
        #file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/representative.txt',names=columns)
        #print(file.head())
        #print("This might a take while..")
        Data=file['Data']
        Data=np.array(Data)

        _centrality_array = dgl.ndarray.array(Data)
        #print("compute array",_computed_array)
        #print(Data.shape)
        #loaded_array = np.load('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster_' + fan + '/' + dataset_name + '_representative.npy')
        #loaded_array = np.load('/data/Framework/graphsage/POP_FINAL' + '/' + dataset_name + '_degree_centrality.txt', names=columns)
        #_centrality_array = dgl.ndarray.array(loaded_array)
        #print("compute array",_computed_array)

        #_computed_array = cp.asarray(_computed_array)
        # _computed_array = _computed_array.to(device)
        # Convert NumPy array to DGL tensor
        # _computed_array = dgl.tensor(_computed_array)
        # device = "cuda" if dgl.cuda.is_available() else "cpu"
        # _computed_array = _computed_array.to(device)
        # _computed_array = _computed_array.tolist
        #print("Array computation done and passed to neighbour.py line 631")
    #else:
        #numpy_array = _computed_array.asnumpy()
        #numpy_array = numpy_array[:-1]
        #numpy_array = np.append(numpy_array, 1)
        #_computed_array = dgl.ndarray.array(numpy_array)
        #print("compute array",_computed_array)
    return _centrality_array

def cluster_formation(g):
    node_feature=g.ndata["feat"]
    node_label=g.ndata["label"]
    print("node features:",node_feature)
    print("node labels:",node_label)
    '''
    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
    '''
    #start=time.time()
    f=g.ndata["feat"].shape[1]
    #print("f={}".format(f))
    #finding nodes with feature vectors of all zeros.
    zeros_tensor=torch.zeros(f)
    equality_mask = torch.all(node_feature == zeros_tensor, dim=1)

    # Find the index where all elements are zero
    index = torch.nonzero(equality_mask, as_tuple=True)[0]
    #print("Number of feature vector of zeros is {}".format(len(index)))

    #print("Index of 1D tensor in 2D tensor where all elements are zero:", index)
    #print(len(index))
    #print(type(index))
    file_path3 = 'index.pt'
    torch.save(index, file_path3)

    index=index.tolist()
    #file_path3 = 'index.txt'

    #finding  distinct node for each cluster
    x = torch.empty(0, f)
    p=800
    for i in range(p):
         x = torch.cat((x,node_feature[i].view(1, -1)), dim=0)
    data_normalized = F.normalize(x, p=2, dim=1)
    #Calculate the cosine similarity matrix
    similarity_matrix = torch.mm(data_normalized, data_normalized.t())
    my_list = []
    rows,columns = similarity_matrix.shape
    for i in range(rows):
        for j in range(columns):
            element = similarity_matrix[i,j]
            if (i < j):
                my_list.append(element)
    sorted_list = sorted(my_list)

    # Get the least 20 elements for 20 clusters
    least_k_elements = sorted_list[:20]
    least_k_elements = torch.tensor(least_k_elements)
    #print("Least similarity")
    #print(least_k_elements)
    #x = torch.empty(0, 2)
    #x = torch.zeros(2,dtype=torch.int64)
    x=torch.Tensor([])
    #x = torch.empty((), dtype=torch.int64)
    #print("x:",x)
    for i in least_k_elements:
        indices = torch.where(similarity_matrix == i)
        #print(indices)
        #print(type(indices))
        merge=torch.cat((indices[0],indices[1]),dim=0)
        #print("merge",merge)
        #x = torch.cat((x,indices[0].view(1, -1)), dim=0)
        x = torch.cat((x,merge), dim=0)
        #print("indices of {} is".format(i))
        #print(indices)
        #print(type(indices))
    #print("index of Least similar nodes")
    #indices_to_remove = [0,1]
    #x = torch.tensor([x[i] for i in range(len(x)) if i not in indices_to_remove])
    x = torch.unique(x)
    x = x.type(torch.int64)
    #print("x:",x)
    y=x[ :20]
    #print("y",y)
    x=node_feature[y]
    clusters = y.view(-1, 1)
    #print("initial cluster")
    #print(y)
    clusters=clusters.tolist()
    #x=node_feature[[4,9,33,44,67]]
    #print("node feature of 5 nodes")
    #print(x)
    #clusters=[[4],[9],[33],[44],[67]]
    cluster_id=torch.Tensor([])
    cluster_id = cluster_id.to(dtype=torch.int64)
    for i in range(g.num_nodes()):    
        if i not in index:
            #for i in range(50):
            similarities = torch.tensor([F.cosine_similarity(node_feature[i],tensor, dim=0) for tensor in x])
            #print("similarity vector")
            #print(similarities)
            largest_element = torch.max(similarities)
            indices = torch.where(similarities == largest_element)
            #print("indices",indices)
            #ii,=indices
            ii=indices[0]
            if(len(ii)>1): #more than 1 elements has same higher similarity value
                cluster_id=torch.cat((cluster_id,torch.tensor([ii[0]])))
                if i not in clusters[ii[0]]:
                   clusters[ii[0]].append(i)
            else:       
                cluster_id=torch.cat((cluster_id,ii))
            #cluster_id[i]=ii
                if i not in clusters[ii]:
                     clusters[ii].append(i)
            #print("cluster forming")     
            #print("Similarity of node {} with respect to 5 cluster are".format(i))
            #print(similarities)
            #print("highest similarity is {}".format(largest_element))
            #print("index is {}".format(ii))
            #print(type(ii))
            #for j in clusters:
                #print(j)
        else:
            cluster_id=torch.cat((cluster_id,torch.tensor([-1]))) #cluster is -1 for nodes having feature vector of all zeros.
    #for j in clusters:
        #print(j)
    '''    
    file_path = 'clusters.txt'

    # Open the file for writing and save the 2D list
    with open(file_path, 'w') as file:
        for row in clusters:
            file.write(' '.join(map(str, row)) + '\n')

    cluster_id = cluster_id.to(dtype=torch.int64)
    cluster_id = cluster_id.tolist()
    #file_path1 = 'cluster_id.pt'
    file_path1 = 'cluster_id.txt'
    with open(file_path1, "w") as file:
        for value in cluster_id:
            file.write(f"{value}\n")
    # Save the tensor to a file
    #torch.save(cluster_id, file_path1)
    # Load the tensor from the file
    #loaded_tensor = torch.load(file_path1)
    #print("cluster id")
    #print(cluster_id)
    #print("#" * 120)
    #FINDING THE REPRESENTATIVE FOR EACH CLUSTER
    representative = torch.empty(0, f)
    for j, row in enumerate(clusters):
        #print(j)
        y=node_feature[row]
        summ=torch.zeros(f)
        for p in y:
            summ+=p
        r=summ/len(y)
        representative = torch.cat((representative, r.view(1, -1)), dim=0)
    #PRINTING REPRESENTIVE OF EACH CLUSTER
    #for j, row in enumerate(representative):
        #print("representative of {} is".format(j))
        #print(row)
    file_path2 = 'representative.pt'
    torch.save(representative, file_path2)
    #print("Time for clustering is {}".format(time.time()-start))
    '''
    # Convert tensor to DGL NDArray
    dgl_cluster_id = dgl.ndarray.array(cluster_id)
    print("cluster_id")
    print(dgl_cluster_id)
    return dgl_cluster_id
    
def get_part_array(G, dataset_name=None, fan=None):
    # print("array passed")
    return metis_partition(G, dataset_name, fan)
    #return cluster_formation(G)

