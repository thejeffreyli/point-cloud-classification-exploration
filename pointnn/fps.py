import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Let's create some sample data for demonstration purposes
# Replace this with your actual data
labels = np.load('res/labels.npy')
fps_0 = np.load('res/xyz0.npy')
fps_1 = np.load('res/lc_xyz0.npy')
fps_2 = np.load('res/lc_xyz1.npy')
fps_3 = np.load('res/lc_xyz2.npy')
fps_4 = np.load('res/lc_xyz3.npy')
point_clouds = [fps_0, fps_1, fps_2, fps_3, fps_4]
names = ["chair", "vase", "lamp"]
for i in range(8):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})
    #ax = fig.add_subplot(151, projection='3d')
    for j in range(5):       
        axs[j].scatter(point_clouds[j][i,:,2], point_clouds[j][i,:,0], point_clouds[j][i,:,1], s=0.8)
            
        axs[j].set_xlabel('X Label')
        axs[j].set_ylabel('Y Label')
        axs[j].set_zlabel('Z Label')
        if j == 0:
            axs[j].set_title(names[labels[i]])        
    plt.show()
