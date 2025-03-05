from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
# Scale and visualize the embedding vectors
def plot_embedding(X,Y, image, scale=0.005, title=None, outfile=None, fmt=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0) #normalization
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(16, 16))
    colors = ['blue', 'red','yellow', 'black', 'purple', 'lime', 'cyan', 'orange', 'gray','green','lightblue','navy', 'turquoise', 'darkorange', 'whitesmoke']

    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                 color=str(colors[Y[i]]),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < scale:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            im = offsetbox.OffsetImage(image[i][::-1,:,2], cmap='jet',zoom=0.6)
            imagebox = offsetbox.AnnotationBbox(
                im,X[i])
            ax.add_artist(imagebox)
    #plt.xticks([]), plt.yticks([])
    plt.grid()
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    if outfile:
        if fmt:
            plt.savefig(outfile, bbox_inches='tight', format=fmt)
        else:
            plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    plt.close()


def Visualization_2Ddistribution(data, label, n_clusters, mode="class", title=None, fmt=None, outfile=None, labels_fig = None):
    if mode == "class" :
        colors = ['white', 'blue', 'red','green','yellow','purple', 'lime', 'cyan', 'black', 'orange', 'gray', 'lightgreen', 'lightblue', 'navy', 'turquoise', 'violet', 'whitesmoke', 'rosybrown', 'darkgreen', 'crimson']
        colors = [
            'white', 'blue', 'red', 'green', 'yellow', 'purple', 'lime', 'cyan', 'black',
            'orange', 'gray', 'lightgreen', 'lightblue', 'navy', 'turquoise', 'violet',
            'whitesmoke', 'rosybrown', 'darkgreen', 'crimson', 'magenta', 'olive', 'brown',
            'maroon', 'gold', 'silver', 'pink', 'orchid', 'indigo', 'khaki'
        ]
    elif mode == "cluster" :
        colors = ['blue', 'red', 'yellow', 'purple', 'lime', 'cyan', 'black', 'orange', 'gray', 'lightgreen', 'lightblue', 'navy', 'turquoise', 'violet', 'whitesmoke', 'rosybrown', 'darkgreen', 'crimson']
        colors = [
            'white', 'blue', 'red', 'green', 'yellow', 'purple', 'lime', 'cyan', 'black',
            'orange', 'gray', 'lightgreen', 'lightblue', 'navy', 'turquoise', 'violet',
            'whitesmoke', 'rosybrown', 'darkgreen', 'crimson', 'magenta', 'olive', 'brown',
            'maroon', 'gold', 'silver', 'pink', 'orchid', 'indigo', 'khaki'
        ]
    # 在 1x1 的網格上繪製子圖形
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cen = int(n_clusters)
    # 調整圖形的外觀
    for i in range(int(max(label)+1)):
        x = data[:-cen, 0][label == i] 
        y = data[:-cen, 1][label == i] 
        ax.scatter(x, y,s =10,  c=colors[i] ,marker='o', edgecolors='black')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #centers
    x1 = data[-cen:, 0] 
    y1 = data[-cen:, 1]
    ax.scatter(x1, y1,s=100, alpha=1, linewidths=1.5, c='y' ,marker='*', edgecolors='black')
              
    if mode == "class" :
        plt.legend(['DebrisFlow','Earthquake', 'Noise', 'Rockfall','Centroid'])
    elif mode == "cluster" :
        if labels_fig:
            plt.legend(labels_fig, borderaxespad=0.)
        else:
            num = ['Cluster-'+str(i) for i in range(cen)]
            num.append('Centroid')
            plt.legend(num, borderaxespad=0.)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
              
    if outfile:
        if fmt:
            fig.savefig(outfile, bbox_inches='tight', format=fmt)
        else:
            fig.savefig(outfile, bbox_inches='tight')
              
    # plt.show()
    plt.close(fig)

def cluster_assignments(data,n_clusters, savename):
    clusters = ["cluster"+str(i) for i in range(n_clusters)]
    num = [np.sum(data==i) for i in range(n_clusters)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(clusters, num, tick_label=clusters, width=0.55, color='slategray')
    ax.set_title('Class Assignments')        
    #ax.set_xlabel('Class')                             
    ax.set_ylabel('Event')
    rect = ax.patches
    for rect, num  in zip(rect, num ):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, num,
                ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(savename,bbox_inches='tight', transparent=False)
    # plt.show()
    plt.close()