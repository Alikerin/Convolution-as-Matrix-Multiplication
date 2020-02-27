# Convolution-as-Matrix-Multiplication
This repository contains two different implementations of 2D Convolution for 3-Channel Images using Matrix Multiplication

The first implementation which was motivated by this article: https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544 was obtained by extracting the image patch that corresponds to each filter in the image and flattening it into a column vector. These column vectors make up the matrix of patches. The matrix of patches for each image channel is stacked together vertically to make the complete matrix of patches for a multi-channel image. Similarly, the kernel is flattened into a row vector and the kernels for a multi-channel image are stacked together horizontally to form the kernel matrix. The matrix multiplication of the matrix of patches and kernel matrix result in a convolution of the image and the kernel.

In the second implementation which was motivated by this article: https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0 is performed as a dot product of the image with a pseudo-image where every other pixel value is zero except for the pixels that corresponds to the kernel at position i (where i is the integer index of elements in the resulting feature map). These pseudo-images are flattened into a row vector and stacked vertically together to form the kernel matrix. The image is flattened into a column vector and multiplied by the kernel matrix. The result is the convolution of the image and the kernel. 

The time complexity of both implementations are approximately:
1. O(no_ch*fmap_size^2)
2. O(fmap_size^2)
where no_ch is the number of channels, which is 3 for RGB image_size and fmap_size is the feature map size
