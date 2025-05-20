from torch_geometric.data import Data
import torch
import numpy as np
import os
from ESM_2 import all_seq_embeddings
from torch_geometric.loader import DataLoader
test_embeddings = all_seq_embeddings
graph_list = []

for idx in range(len(test_embeddings)):
    emb = torch.tensor(test_embeddings[idx], dtype=torch.float)  # (L, 320)
    L = emb.size(0)

    adj_path = f"distance_test_matrices/Sequence_{idx+1}.npy"
    if not os.path.exists(adj_path):
        print(f"Missing: {adj_path}")
        continue
    adj = np.load(adj_path)

    edge_index = []
    for i in range(L):
        for j in range(L):
            if adj[i][j] != 0:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=emb, edge_index=edge_index)
    graph_list.append(data)
test_loader = DataLoader(graph_list, batch_size=1, shuffle=False)