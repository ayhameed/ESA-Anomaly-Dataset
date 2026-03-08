import numpy as np

data = np.load("telecommand_3", allow_pickle=True)
print(type(data))
print(data.shape)
print(data[:10])