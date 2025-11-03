from dgl.data import RedditDataset

# Automatically downloads to ~/.dgl/
dataset = RedditDataset(self_loop=True)
graph = dataset[0]

print("Reddit dataset downloaded!")
print(f"Graph: {graph}")
print(f"Number of nodes: {graph.num_nodes()}")
print(f"Number of edges: {graph.num_edges()}")

