import numpy as np

# 載入 .npy 檔案
data = np.load("Sequence_1.npy")

# 查看資料形狀
print("Shape:", data.shape)

# 查看前幾筆資料
print("First few entries:", data)  # 根據維度調整 slicing