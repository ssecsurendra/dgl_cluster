import numpy as np

graph_npz = "/home/surendra/.dgl/reddit_extracted/reddit_graph.npz"
data_npz = "/home/surendra/.dgl/reddit_extracted/reddit_data.npz"

graph = np.load(graph_npz)
data = np.load(data_npz)

print("📦 reddit_graph.npz keys:", list(graph.keys()))
print("📦 reddit_data.npz keys:", list(data.keys()))

