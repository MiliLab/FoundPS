import numpy as np   
from scipy import ndimage
import cv2
import math
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
 

def RSGenerate(image_ori, NUM_BAND, percent = 1, colorization=True): 
    image_ori = image_ori.astype(np.float64); 
    
    if NUM_BAND == 4:
        subset = [2, 1, 0]
    elif NUM_BAND == 7:
        subset = [2, 1, 0]  
    elif NUM_BAND == 8:
            subset = [4, 2, 1]
    elif NUM_BAND == 10:
            subset = [3, 2, 1]
    else:
        exit('Unsupported NUM_BAND = %d', NUM_BAND)

    image = image_ori[:,:,subset]

    m, n, c = image.shape 
    image_normalize = image / np.max(image)
    image_generate = np.zeros(list(image_normalize.shape))
    if colorization: 
        for i in range(c):
            image_slice = image_normalize[:, :, i]
            pixelset = np.sort(image_slice.reshape([m * n]))
            maximum = pixelset[np.floor(m * n * (1 - percent / 100)).astype(np.int32)]
            minimum = pixelset[np.ceil(m * n * percent / 100).astype(np.int32)]
            image_generate[:, :, i] = (image_slice - minimum) / (maximum - minimum + 1e-9)
            pass
        image_generate[np.where(image_generate < 0)] = 0
        image_generate[np.where(image_generate > 1)] = 1
        image_generate = cv2.normalize(image_generate, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image_generate.astype(np.uint8)


def tSNE(data,color,n_components,save_path):
    fig = plt.figure(figsize=(8, 8))		# 指定图像的宽和高
    plt.suptitle("Dimensionality Reduction and Visualization of S-Curve Data ", fontsize=14)		# 自定义图像名称
    # t-SNE的降维与可视化
    ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(data)
    ax1 = fig.add_subplot(2, 1, 2)
    plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax1.set_title('t-SNE Curve', fontsize=14)
    # 显示图像
    plt.savefig("tSNE.png", dpi=500)