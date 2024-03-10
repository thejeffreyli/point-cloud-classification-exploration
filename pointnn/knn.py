import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Let's create some sample data for demonstration purposes
# Replace this with your actual data
labels = np.load('res/labels.npy')
xyz = np.load('res/xyz2.npy')
lc_xyz = np.load('res/lc_xyz2.npy')
knn = np.load('res/knn_2.npy')
names = ["chair", "vase", "lamp"]
import pdb
pdb.set_trace()
num_points = xyz[0, :, 2].shape[0]
for i in range(8):
    colors = np.array([(0, 0, 1, 0.3)] * num_points)
    fig, axs = plt.subplots(1, 3, figsize=(25, 5), subplot_kw={'projection': '3d'})
    #ax = fig.add_subplot(151, projection='3d')
    indices_list = []
    for j in range(90):
        target_value = knn[i,0,j,:]
        indices = np.where((xyz[i,:,:] == target_value).all(axis=1))[0]
        indices_list.append(indices)
    indices = np.stack(indices_list)
    print(len(indices))    
    axs[0].scatter(xyz[i,:,2], xyz[i,:,0], xyz[i,:,1], s=1.0, color=colors)
    axs[1].scatter(lc_xyz[i,:,2], lc_xyz[i,:,0], lc_xyz[i,:,1], s=1.0, color=(0, 0, 1, 0.3))
    colors[indices] = (1, 0, 0, 1) 
    axs[2].scatter(xyz[i,:,2], xyz[i,:,0], xyz[i,:,1], s=1.0, color=colors)
    
    for j in range(3):       
        axs[j].set_xlabel('X Label')
        axs[j].set_ylabel('Y Label')
        axs[j].set_zlabel('Z Label')
        if j == 0:
            axs[j].set_title(names[labels[i]])        
    plt.savefig(f'images/knn_{i}.png')
