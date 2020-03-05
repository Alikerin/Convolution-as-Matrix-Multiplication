# @Author: Ibrahim Salihu Yusuf <yusuf>
# @Date:   2020-02-26T21:05:43+02:00
# @Email:  sibrahim1396@gmail.com
# @Project: Audio Classifier
# @Last modified by:   yusuf
# @Last modified time: 2020-03-05T15:59:44+02:00



# 2D 3-Channel Convolution
import numpy as np

#implementation of https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544
def convolution1(image, kernel, stride=1, padding=0):
    channels = image.shape[0]
    no_kernels = kernel.shape[0]
    kernel_width = kernel.shape[3]
    kernel_height = kernel.shape[2]
    image_height = image.shape[1]
    image_width = image.shape[2]
    fmap_h = int((image_height-kernel_height+(2*padding))/stride)+1
    fmap_w = int((image_width-kernel_width+(2*padding))/stride)+1
    new_image = np.zeros((channels*kernel_height*kernel_width, fmap_h*fmap_w))
    fmaps = []
    if padding>0:
        temp = np.zeros((channels,image_height+2*padding, image_width+2*padding))
        temp[:,padding:-padding,padding:-padding] = image
        image = temp
        image_height = image.shape[1]
        image_width = image.shape[2]
        del temp
    idx, ch = 0,0
    while ch<channels:
        row,idx = 0,0
        while row+kernel_height <= image_height:
            col=0
            while col+kernel_width <= image_width:
                new_image[kernel_height*kernel_width*ch:kernel_height*kernel_width*ch+(kernel_height*kernel_width), idx] = (image[ch,row:row+kernel_height,col:col+kernel_width]).reshape(-1)
                idx+=1
                col+=stride
            row+=stride
        ch+=1
    for k in range(no_kernels):
        fmap = (kernel[k].reshape(-1) @ new_image).reshape(fmap_h,fmap_w).astype(np.int)
        fmaps.append(fmap)
    return np.array(fmaps)

#An implementation of https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0
def convolution2(image, kernel, stride=1, padding=0):
    channels = image.shape[0]
    kernel_width = kernel.shape[3]
    kernel_height = kernel.shape[2]
    no_kernels = kernel.shape[0]
    image_height = image.shape[1]
    image_width = image.shape[2]
    fmap_h = int((image_height-kernel_height+(2*padding))/stride)+1
    fmap_w = int((image_width-kernel_width+(2*padding))/stride)+1
    fmaps=[]
    if padding>0:
        temp = np.zeros((channels,image_height+2*padding,image_width+2*padding))
        temp[:,padding:-padding,padding:-padding] = image
        image = temp
        image_height = image.shape[1]
        image_width = image.shape[2]
        del temp
    for k in range(no_kernels):
        row, col = 0,0
        conv_matrix = np.zeros((fmap_h*fmap_w, image_height*image_width*channels))
        idx=0
        while row+kernel_height <= image_height:
            col=0
            while col+kernel_width<=image_width:
                temp_kernel = np.zeros_like(image)
                temp_kernel[:,row:row+kernel_height, col:col+kernel_width] = kernel[k]
                conv_matrix[idx] = temp_kernel.reshape(-1)
                col+=stride
                idx+=1
            row+=stride
        fmap = (np.array(conv_matrix) @ image.reshape(-1)).reshape(fmap_h,fmap_w).astype(np.int)
        fmaps.append(fmap)
    return np.array(fmaps)
