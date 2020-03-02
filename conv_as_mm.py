# @Author: Ibrahim Salihu Yusuf <yusuf>
# @Date:   2020-02-26T21:05:43+02:00
# @Email:  sibrahim1396@gmail.com
# @Project: Audio Classifier
# @Last modified by:   yusuf
# @Last modified time: 2020-03-02T09:34:58+02:00



# 2D 3-Channel Convolution
import numpy as np

#implementation of https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544
def convolution1(image, kernel, stride=1, padding=0):
    channels = image.shape[0]
    no_kernels = kernel.shape[0]
    kernel_size = kernel.shape[1]
    image_size = image.shape[1]
    fmap_size = int((image_size-kernel_size+(2*padding))/stride)+1
    new_image = np.zeros((channels*kernel_size*kernel_size, fmap_size*fmap_size))
    fmaps = []
    if padding>0:
        temp = np.zeros((channels,image_size+2*padding,image_size+2*padding))
        temp[:,padding:-padding,padding:-padding] = image
        image = temp
        image_size = image.shape[1]
        del temp
    idx, ch = 0,0
    while ch<channels:
        row,idx = 0,0
        while row+kernel_size <= image_size:
            col=0
            while col+kernel_size <= image_size:
                new_image[kernel_size*kernel_size*ch:kernel_size*kernel_size*ch+(kernel_size*kernel_size), idx] = (image[ch,row:row+kernel_size,col:col+kernel_size]).reshape(-1)
                idx+=1
                col+=stride
            row+=stride
        ch+=1
    for k in range(no_kernels):
        fmap = (kernel[k].reshape(-1) @ new_image).reshape(fmap_size,fmap_size).astype(np.int)
        fmaps.append(fmap)
    return fmaps

#An implementation of https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0
def convolution2(image, kernel, stride=1, padding=0):
    channels = image.shape[0]
    kernel_size = kernel.shape[1]
    no_kernels = kernel.shape[0]
    image_size = image.shape[1]
    fmap_size = int((image_size-kernel_size+(2*padding))/stride)+1
    fmaps=[]
    if padding>0:
        temp = np.zeros((channels,image_size+2*padding,image_size+2*padding))
        temp[:,padding:-padding,padding:-padding] = image
        image = temp
        image_size = image.shape[1]
        del temp
    for k in range(no_kernels):
        row, col = 0,0
        conv_matrix = np.zeros((fmap_size*fmap_size, image_size*image_size*channels))
        idx=0
        while row+kernel_size <= image_size:
            col=0
            while col+kernel_size<=image_size:
                temp_kernel = np.zeros_like(image)
                temp_kernel[:,row:row+kernel_size, col:col+kernel_size] = kernel[k]
                conv_matrix[idx] = temp_kernel.reshape(-1)
                col+=stride
                idx+=1
            row+=stride
        fmap = (np.array(conv_matrix) @ image.reshape(-1)).reshape(fmap_size,fmap_size).astype(np.int)
        fmaps.append(fmap)
    return np.array(fmaps)
