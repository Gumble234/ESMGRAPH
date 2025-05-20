import torch
from torch_geometric.data import Data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 讀取生成的圖檔
graph = torch.load('generated_graphs/graph_1.pt')

print(graph)  # 會顯示節點數、邊數、屬性維度等基本資訊

# 印出節點特徵大小和部分內容
print("Node feature shape:", graph.x.shape)
print("Node features (前5筆):", graph.x[:5])

# 印出邊的索引（source -> target）
print("Edge index shape:", graph.edge_index.shape)
print("Edge index:", graph.edge_index)

# 你也可以用 NetworkX 或 matplotlib 視覺化這個圖
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

G = to_networkx(graph, to_undirected=True)
nx.draw(G, with_labels=True)
plt.show()
