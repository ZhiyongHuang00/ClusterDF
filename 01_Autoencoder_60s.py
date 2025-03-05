# -*- coding: utf-8 -*-
'''
@Time    : 2024/9/6 10:59
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 01_Autoencoder_60s.py
@Function: {}:
'''
# !/usr/bin/env python
# coding: utf-8
from datetime import datetime
import pandas as pd
# 打印当前的日期和时间
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
import Model_60s
import glob
from plot_distribution import plot_embedding, Visualization_2Ddistribution, cluster_assignments
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import trange

sns.set_style('darkgrid')
sns.set_palette('muted')
print(tf.config.list_physical_devices('GPU'))

print(tf.config.list_physical_devices())
print(tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

save_dir = "./STFT/Z_stft"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# 获取当前的日期和时间
current_datetime = datetime.now()
current_datetime = current_datetime.strftime("%Y%m%d%H%M%S")


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
        # data = data[:1000]
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
# train_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_aug_40\maxmin_40s\*.npy")
# label_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_aug_40\maxmin_40s\*\*.npy")
# train_files = glob.glob(r"D:\Pycharm_Project\Classification\Wavelet\tmp_std_40s\*.npy")
# label_files = glob.glob(r"D:\Pycharm_Project\Classification\Wavelet\tmp_std_40s\label\*.npy")
train_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_60\aug_maxmin\*.npy")
label_files = glob.glob(r"D:\Pycharm_Project\Classification\STFT\stft_60\aug_maxmin\*\*.npy")
x_train = load_and_concatenate(train_files)
y_train = load_and_concatenate(label_files, 0)
file_position = load_and_concatenate(label_files, 1)
y_train = y_train.astype(np.float32)
x_train = np.expand_dims(x_train, axis=-1)
x_train = x_train.astype(np.float32)
# x_train = tf.convert_to_tensor(x_train, dtype=tf.float16)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float16)
print(x_train.shape)

"""1DF、2EQ、0NS、3RF"""


###make noisy data
def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """
    noise_factor = 0.2
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=3.0, size=array.shape
    )
    return noisy_array

def plot_decoder_output(origianl_data,output_data):
    # Plot the input spectrogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(origianl_data), aspect='auto', cmap='viridis')
    plt.title('Input Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()

    # Plot the output spectrogram
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(output_data), aspect='auto', cmap='viridis')
    plt.title('Output Spectrogram (Autoencoder)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()

    plt.suptitle('Autoencoder Input vs Output Spectrogram')
    plt.tight_layout()
    plt.savefig(r"D:\Desktop\aiting.png")
    # Save the figure
    plt.show()
    plt.close()
# x_train_noisy = noise(x_train)

###building model
encoder, decoder, autoencoder = Model_60s.CAE(input_shape=(64, 600, 1), filters=[12, 24, 36, 48, 64], encoding_filters=16,
                                            set_seed=[False, 87], summary=True)

###training 
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.compile(optimizer='adam', loss='mse')
csv_logger = tf.keras.callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)# min_delta=1e-5,

history = autoencoder.fit(x_train, x_train,
                          batch_size=256,
                          epochs=2000,
                          callbacks=[csv_logger, early_stopping]
                          )

loss_values = history.history['loss']
save_loss(loss_values)
output_data = autoencoder.predict(x_train)

# for i in trange(0,100000):
#     i = 10000
#     plot_decoder_output(x_train[i,:,:],output_data[i,:,:])

# WeightsFileName =  ['D:\Pycharm_Project\Classification\models_save/Pretrain_encoder.h5',
#                     'D:\Pycharm_Project\Classification\models_save/Pretrain_autoencoder.h5',
#                     'D:\Pycharm_Project\Classification\models_save/Pretrain_decoder.h5']
WeightsFileName = [f'D:\Pycharm_Project\Classification\clustering_method\DEC\Model_output//{current_datetime}ACE_Pretrain_encoder.h5',
                   f'D:\Pycharm_Project\Classification\clustering_method\DEC\Model_output//{current_datetime}ACE_Pretrain_autoencoder.h5',
                   f'D:\Pycharm_Project\Classification\clustering_method\DEC\Model_output//{current_datetime}ACE_Pretrain_decoder.h5']
encoder.save_weights(WeightsFileName[0])
autoencoder.save_weights(WeightsFileName[1])
decoder.save_weights(WeightsFileName[2])
features = encoder.predict(x_train)

print('feature shape=', features.shape)
import metrics

metrics_list_pretrain = []
# features = encoder.predict(x_train)

for i in trange(2, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=30).fit(features)
    y_pred = kmeans.predict(features)
    truelist, predlist, predlist_km = metrics.tracker(y_train, y_pred, n_clusters=i)
    y_pred_last = predlist
    nmi = metrics.nmi(truelist, predlist)
    ari = metrics.ari(truelist, predlist)
    purity = metrics.purity_score(truelist, predlist)

    print('nmi={0},ari={1},purity={2}'.format(nmi, ari, purity))
    metric = [i, nmi, ari, purity]
    metrics_list_pretrain.append(metric)
    del y_pred
# Find the i corresponding to the maximum purity
max_purity_metric = max(metrics_list_pretrain, key=lambda x: x[3])
n_clusters = max_purity_metric[0]

print(metrics_list_pretrain)
# # Evalutation
# sbhzy
# In[3]:
plt.figure(figsize=(8, 6))
nmi = [i[1] for i in metrics_list_pretrain]
ari = [i[2] for i in metrics_list_pretrain]
purity = [i[3] for i in metrics_list_pretrain]
clusters = [i[0] for i in metrics_list_pretrain]

plt.plot(clusters, nmi, 'o-', color='red', linewidth=2)
plt.plot(clusters, ari, 'o-', color='g', linewidth=2)
plt.plot(clusters, purity, 'o-', color='steelblue', linewidth=2)

plt.title('Number of clusters')
plt.legend()
plt.xlabel('Number of clusters')
plt.ylabel('score')
plt.legend(["nmi", "ari", "purity"])
plt.ylim([0.2, 1.0])
plt.yticks()
# plt.show()
plt.savefig(f"D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}DEC_Number of clusters",
            bbox_inches='tight', transparent=False)
print("Saved figures one")
# sbhzy
plt.close()
kmeans = KMeans(n_clusters=n_clusters, random_state=87).fit(features)
y_pred = kmeans.fit_predict(features)
y_pred_last = np.copy(y_pred)
# 使用 t-SNE 對 `encodedConv2D_imgs` 資料降維
centers = kmeans.cluster_centers_  # 聚类中心
enc_cen = np.append(features, centers, axis=0)
df_enc_cen = pd.DataFrame(enc_cen)
df_enc_cen.to_csv('D:\Desktop/df_enc_cen.csv', index=False)
print("Saved to_csv one")

redu = TSNE(n_components=2, perplexity=500, max_iter=1000, random_state=87).fit_transform(enc_cen)  # 将数据降到二维，以便于可视化

# 将结果保存到DataFrame
print("TSNE is finished")


y_true = np.ndarray(shape=(len(y_train)), dtype='float32')
for i, data in enumerate(y_train):
    # print (i,data)
    if data == 0:
        y_true[i] = np.array(0)
    elif data == 1:
        y_true[i] = np.array(1)
    elif data == 2:
        y_true[i] = np.array(2)
    elif data == 3:
        y_true[i] = np.array(3)
# 为聚类中心创建标签，赋值为 -1
center_labels = np.full((centers.shape[0],), -1)
# 合并标签
all_labels = np.concatenate((y_pred_last, center_labels))
# In[9]:
df = pd.DataFrame(redu, columns=['Dim1', 'Dim2'])
df['Label'] = all_labels
df.to_csv('D:\Desktop/2D_tsne_results.csv', index=False)

Visualization_2Ddistribution(redu, y_true, n_clusters, mode="class", title="Pretrain Model Feature Domain by t-SNE",
                             fmt=None,
                             outfile=f"D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}DEC_pretrain_model_output")
Visualization_2Ddistribution(redu, y_pred_last, n_clusters, mode="cluster", title=None, fmt=None,
                             outfile=f"D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}DEC_pretrain_model_output_2")
# ## Model Performace
# In[10]:
import metrics

truelist, predlist, predlist_km = metrics.tracker(y_train, y_pred_last, n_clusters=n_clusters)
# 找出错误分类的位置
error_indices = np.where(truelist != predlist)[0]

# 创建一个 DataFrame，包含位置、正确标签和错误标签
error_data = {
    '位置': error_indices,
    '正确标签': truelist[error_indices],
    '错误标签': predlist[error_indices],
    "文件名称": file_position[error_indices]
}
error_df = pd.DataFrame(error_data)

# 保存为 Excel 文件
error_df.to_excel(
    f'D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}DEC_error_classification.xlsx',
    index=False)
mat = confusion_matrix(truelist, predlist)
accuracy = np.trace(mat) / np.sum(mat)
print("Accuracy is {}".format(accuracy))
"""1DF、2EQ、0NS、3RF"""

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, xticklabels=['NS', 'DF', 'EQ', 'RF'],
            yticklabels=['NS', 'DF', 'EQ', 'RF']
            )
plt.title("Pretrain model confusion matrix", fontsize=12, fontweight='bold')
plt.xlabel('predicted label')
plt.ylabel('true label')

plt.savefig(
    f"D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}DEC_Pretrain model confusion matrix",
    bbox_inches='tight', transparent=False)
# plt.show()
plt.close()
# In[11]:
print('nmi=', metrics.nmi(truelist, predlist), 'ari=', metrics.ari(truelist, predlist_km))
Purity = []
purity = metrics.purity_score(truelist, predlist)
print("Purity : ", purity)
Purity.append(purity)
# ## Class Assignments

cluster_assignments(y_pred_last, n_clusters,
                    f"D:\Pycharm_Project\Classification\clustering_method\DEC\Savefig\\{current_datetime}DEC_cluster_assignments")
debug = 1

from datetime import datetime

# 获取当前时间
current_time = datetime.now()

# 格式化时间为可读形式
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

# 打印当前时间
save_mode_output = {
    "spectral": x_train,
    'encoder': features,
    "decoder": output_data,
    'redu': redu,
    'metrics_list_pretrain': metrics_list_pretrain,
    "error_df": error_df,
    "mat": mat,
    "accuracy": accuracy,
    "nmi": metrics.nmi(truelist, predlist),
    "ari": metrics.ari(truelist, predlist_km),
    "purity": purity,
    "truelist": truelist,
    "y_pred_last": y_pred_last,
    "n_clusters": n_clusters,

}
save_name_oup = f'D:\\Pycharm_Project\\Classification\\clustering_method\\DEC\\Model_output\\{current_datetime}_{n_clusters}_{purity}_ACE_model.npz'
np.savez(save_name_oup, **save_mode_output)
print('save to file：' , save_name_oup)
print("当前完成时间:", formatted_time)

# In[ ]:
a = 1


