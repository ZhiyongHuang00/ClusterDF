#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date
today = str(date.today())
print("Today's date:", today)
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import Model_60s
from plot_distribution import plot_embedding,Visualization_2Ddistribution,cluster_assignments
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import metrics 
import seaborn as sns
import csv
import glob
import pandas as pd
from datetime import datetime

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, \
    UpSampling1D, Flatten, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, \
    CSVLogger
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sns.set_style('darkgrid')
sns.set_palette('dark')
print (tf.config.list_physical_devices())
print (tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   #Invalid device or cannot modify virtual devices once initialized.
#   pass

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 获取GPU设备

# 如果存在GPU设备，设置内存增长
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print("Error setting memory growth:", e)
# In[2]:
current_datetime = datetime.now()
current_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
#Setting Config
config = {"dirname" : "iter3e4",
          "WeightsFileName" : [r".\Model_output\20241016221158ACE_Pretrain_encoder.h5",
                               r".\Model_output\20241016221158ACE_Pretrain_autoencoder.h5",
                               r".\Model_output\20241016221158ACE_Pretrain_decoder.h5"],
          "n_clusters" : 12,
            "gamma" : 0.1,
            "batch_size" : 256,
            "maxiter" : 10000,
            "tol" : 0.001, # tolerance threshold to stop training
            "update_interval" : 10,
            "save_interval" : 5000
}

def load_and_concatenate(files, column_index=None):
    data_list = []
    i = 0
    for file in files:
        data = np.load(file, allow_pickle=True)
        if column_index is not None:
            data = data[:, column_index]

        # if len(data) > 9000:
        #     data = data[5000:10000]
        # else:
        #     data = data[:5000]
        data = data[:3000]
        data_list.append(data)
        print(file)
        del data
        i = i + 1
    return np.concatenate(data_list, axis=0)
def save_loss(loss):
    # 保存损失值到CSV文件
    loss_df = pd.DataFrame(loss, columns=['Loss'])
    loss_df.to_csv(f'D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig//{current_datetime}_Loss.csv', index=False)
    # 绘制损失曲线D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}Number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}_Loss.png')
    # plt.show()
    plt.close()

# 定义文件夹路径
# train_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_40\tmp_std_40s\*.npy")
# label_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_40\tmp_std_40s\*\*.npy")
train_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_60\aug_maxmin\*.npy")
label_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_60\aug_maxmin\*\*.npy")
x_train = load_and_concatenate(train_files)
y_train = load_and_concatenate(label_files, 0)
file_position = load_and_concatenate(label_files, 1)
y_train = y_train.astype(np.float16)
x_train = np.expand_dims(x_train, axis=-1)
x_train = x_train.astype(np.float16)


y_true =  y_train
DEC_weightsdir = config["dirname"]

n_clusters = config["n_clusters"]
save_dir = r'D:\Pycharm_Project\Classification\clustering_method\DEC\DEC_save/weights_{0}_1_{1}/cluster{2}'.format(str(config["gamma"]).split('.')[1],config["dirname"],str(config["n_clusters"]))

if not os.path.exists(save_dir+"/image/"):
    os.makedirs(save_dir+"/image/")
save_imagedir = save_dir+"/image/"

print (save_dir)
print (x_train.shape)


# # Buliding the model
#Buliding the autoencoder
encoder,decoder,autoencoder = Model_60s.CAE(input_shape=(64, 600, 1), filters=[12, 24, 36, 48, 64], encoding_filters=8, summary=True)
encoder.load_weights( config["WeightsFileName"][0])
autoencoder.load_weights( config["WeightsFileName"][1])
decoder.load_weights( config["WeightsFileName"][2])


# ## adding the clustering layer into the bottelneck layer

# In[5]:

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
print('...Finetuning...')
# Define DCEC model
# clustering_layer = Model_60s.ClusteringLayer(config["n_clusters"], name='clustering')(encoder.output)
clustering_layer = ClusteringLayer(config["n_clusters"], name='clustering')(encoder.output)
DECmodel = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output], name='DEC')
DECmodel.compile(loss=['kld', 'mse'], loss_weights=[config["gamma"], 1], optimizer='adam')
DECmodel.summary()

# ## initializing the weights
### initializing the weights using Kmean and assigning them to the model
features = encoder.predict(x_train)
kmeans = KMeans(n_clusters=config["n_clusters"], n_init=20)
y_pred = kmeans.fit_predict(features)
y_pred_last = np.copy(y_pred)
DECmodel.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

## parameters for the finetuning
loss = [0, 0, 0]
index = 0
index_array = np.arange(x_train.shape[0])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
###############################################################################
### simultaneous optimization and clustering
logfile = open(save_dir + '/dcec_log.csv', 'w')
logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'purity', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
logwriter.writeheader()
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
# Initialize variables to store the best purity and corresponding lists
best_purity = 0
best_truelist = None
best_predlist = None
best_predlist_km = None

for ite in range(int(config["maxiter"])):
    if ite % config["update_interval"] == 0:
        if len(x_train) > 15000:
            q1, _ = DECmodel.predict(x_train[0:15000], verbose=0)
            q2, _ = DECmodel.predict(x_train[15000:], verbose=0)
            q = np.concatenate((q1, q2), axis=0)
        else:
            q, _ = DECmodel.predict(x_train, verbose=0)

        # update the auxiliary target distribution p
        p = target_distribution(q)
        # evaluate the clustering performance
        y_pred = q.argmax(1)
        print('Iter %d: ' % (ite), ' ; L = {0} Lc = {1} Lr = {2}'.format(loss[0], loss[1], loss[2]))
        truelist, predlist, predlist_km = metrics.tracker(y_train, y_pred, n_clusters=config["n_clusters"])
        purity = np.round(metrics.purity_score(truelist, predlist_km), 5)
        # Check if current purity is the best so far
        if purity > best_purity:
            best_purity = purity
            best_truelist = truelist.copy()
            best_predlist = predlist.copy()
            best_predlist_km = predlist_km.copy()
        nmi = np.round(metrics.nmi(truelist, predlist_km), 5)
        logdict = dict(iter=ite, purity=purity, nmi=nmi, L=loss[0], Lc=loss[1], Lr=loss[2])
        logwriter.writerow(logdict)
        print('Purity : {0} ,NMI : {1}'.format(purity, nmi))

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < config["tol"]:
            print('delta_label ', delta_label, '< tol ', config["tol"])
            break

    # save intermediate model
    # if ite % config["save_interval"] == 0:
    #     # save DCEC model checkpoints
    #     print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
    #     DECmodel.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

        # IN = encoder.predict(x)
    idx = index_array[index * config["batch_size"]: min((index + 1) * config["batch_size"], x_train.shape[0])]
    loss = DECmodel.train_on_batch(x=x_train[idx], y=[p[idx], x_train[idx]])
    index = index + 1 if (index + 1) * config["batch_size"] <= x_train.shape[0] else 0
    tf.keras.backend.clear_session()
# save DCEC model checkpoints
logfile.close()
print('saving model to:', save_dir + '/dcec_model_final.h5')
DECmodel.save_weights(save_dir + '/dcec_model_final.h5')
print ('finish :'+str(ite))


# In[13]:


import csv
with open(save_dir+ '/dcec_log.csv', newline='') as csvfile:
    purity = []
    rows = csv.DictReader(csvfile)
    for row in rows:
        purity.append(float(row['purity']))
        
Iter = [i*150 for i in range(len(purity))]
values = [0,5000,10000,15000,20000,25000,30000]

plt.figure(figsize=(16,8))
plt.plot(Iter,purity,'bo-', label='Purity', linewidth=2)
plt.title('Purity Score')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.yticks()
plt.savefig(save_imagedir+"/Purity.jpg",dpi=1080,bbox_inches='tight', transparent=False) 
plt.show()
plt.close()


# # load final model_weights

# In[7]:


DECmodel.load_weights(save_dir+ '/dcec_model_final.h5')
print (save_dir+ '/dcec_model_final.h5')
enc = encoder.predict(x_train)


# In[8]:


kmeans = KMeans(n_clusters=config["n_clusters"],n_init=20, random_state=87).fit(enc)
y_pred = kmeans.predict(enc)
y_pred_last = np.copy(y_pred)
centers = kmeans.cluster_centers_
enc_cen = np.append(enc,centers,axis=0)
redu = TSNE(n_components=2,perplexity=40,random_state=74).fit_transform(enc_cen)

truelist,predlist,predlist_km = metrics.tracker(y_train,y_pred_last,n_clusters=config["n_clusters"])
print('nmi=', metrics.nmi(np.array(truelist), np.array(predlist)), 'ari=', metrics.ari(np.array(truelist), np.array(predlist)))
purity = metrics.purity_score(np.array(truelist), np.array(predlist))
print ("Purity : ", purity)


# In[9]:


reconstruction_centers = decoder.predict(centers)
cen = int(len(reconstruction_centers))

# for i,n in enumerate(reconstruction_centers):
#     Sxx = reconstruction_centers[i][...,0]
#     plt.pcolormesh(Sxx ,cmap='jet')
#     plt.axes().get_xaxis().set_visible(False)
#     plt.axes().get_yaxis().set_visible(False)
#     num = str(i)
#     plt.savefig(save_dir+"/image/reconstruction_centers_"+num+"_.jpg",bbox_inches='tight', transparent=False)
#     plt.show()
#     plt.close()


# ## Visualization

# In[17]:


print (os.getcwd())


# In[10]:


y_pred_trans=  np.zeros_like(y_pred_last)
for i,label in enumerate( y_pred_last) :
    if label == 0 :
        y_pred_trans[i] = np.array(0)
    elif label == 1 :
        y_pred_trans[i] = np.array(1)
    elif label == 2 :
        y_pred_trans[i] = np.array(2)
    elif label == 3 :
        y_pred_trans[i] = np.array(3)
    # elif label == 4 :
    #     y_pred_trans[i] = np.array(2)
    # elif label == 5 :
    #     y_pred_trans[i] = np.array(4)


# In[20]:


Visualization_2Ddistribution(redu, y_true, config["n_clusters"], mode="class", title="DECmodel iteration 30000 by t-SNE", fmt=None, outfile=save_imagedir + "DEC_model_output_30000")
Visualization_2Ddistribution(redu, y_pred_trans, config["n_clusters"], mode="cluster", title=None, fmt=None, outfile=save_imagedir + "DEC_model_output_30000_2")


# ## Model Preformance

# In[12]:


truelist,predlist,predlist_km = metrics.tracker(y_train,y_pred_trans,n_clusters=config["n_clusters"])
mat = confusion_matrix(truelist,predlist)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['car','EN','EQ','RF'], yticklabels=['car','EN','EQ','RF']
            )
plt.title("DEC model confusion matrix", fontsize=12, fontweight='bold')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig(save_imagedir+"/DEC model confusion matrix.jpg",bbox_inches='tight', transparent=False) 


# In[13]:


print ("Clustering Accuracy : ",metrics.accuracy_score(np.array(truelist),np.array(predlist)))
print('nmi=', metrics.nmi(np.array(truelist), np.array(predlist)), 'ari=', metrics.ari(np.array(truelist), np.array(predlist)))
Purity = []
purity = metrics.purity_score(np.array(truelist), np.array(predlist))
print ("Purity : ", purity)
Purity.append(purity)


# ## Class Assignments

# In[15]:


# cluster_assignments(y_pred_trans,config["n_clusters"])

# 打印当前时间,
save_mode_output = {
    'best_purity': best_purity,
    'best_truelist': truelist,
    'best_predlist':best_predlist ,
    'best_predlist_km': best_predlist_km,
    "spectral": x_train,
    'encoder': features,
    "Iter": Iter,
    'redu': redu,
    "mat": mat,
    "accuracy": purity,
    "nmi": metrics.nmi(truelist, predlist),
    "ari": metrics.ari(truelist, predlist_km),
    "purity": purity,
    "truelist": truelist,
    "y_pred_last": y_pred_last,
    "n_clusters": n_clusters,

}
save_name_oup = f'D:\\Pycharm_Project\\Classification\\clustering_method\\DEC\\Model_output\\{current_datetime}_{n_clusters}_{purity}_DEC_model.npz'
np.savez(save_name_oup, **save_mode_output)
print('save to file：' , save_name_oup)

# In[ ]:




