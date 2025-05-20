import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from torch_geometric.data import Batch
import torch
from construct_graph import train_loader
from test_graph import  test_loader 
import pandas as pd
class AMP_GCN(torch.nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)  # 0 or 1 分類

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AMP_GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(20):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


model.eval()
predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)[:, 1]  # 取得屬於 class 1 的機率
        predictions.extend(probs.tolist())

result_df = pd.DataFrame({
    "id": [f"Sequence_{i+1}" for i in range(len(predictions))],
    "prediction_score": predictions,
    "predicted_label": [1 if p >= 0.5 else 0 for p in predictions]
})

result_df.to_csv("test_predictions.csv", index=False)
