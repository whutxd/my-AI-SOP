import numpy as np

# 加载NPY文件
data = np.load('lstm/data/npy/y.npy')

# 打印数据类型、形状和内容
print("----类型----")
print(type(data))
print("----形状----")
print(data.shape)
print("----数据----")
print(data)