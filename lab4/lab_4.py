import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.ndimage import correlate, convolve


def ex1_and_2():
    img = skimage.data.chelsea().astype(int)
    gray_img = np.mean(img, axis=2)
    sobel_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    pad_width = sobel_kernel.shape[0] // 2
    padded_img = np.pad(gray_img, pad_width=pad_width, mode='constant')
    
    img_correlated = correlate(padded_img, sobel_kernel)
    img_convolved = convolve(padded_img, sobel_kernel)
    fig, ax = plt.subplots(2, 2, figsize=(10,7))
    ax[0, 0].imshow(img_correlated, cmap='binary_r')
    ax[0, 0].set_title('Correlated - scipy')
    ax[0, 1].imshow(img_convolved, cmap='binary_r')
    ax[0, 1].set_title('Convoluted - scipy')

    img_correlated_out = np.zeros_like(img_correlated)
    img_convolved_out = np.zeros_like(img_convolved)

    for i in range(1, img_correlated.shape[0] - 2):
        for j in range(1, img_correlated.shape[1] - 2):
            img_correlated_out[i, j] = np.sum(
                sobel_kernel * padded_img[i:i+sobel_kernel.shape[0],
                j:j+sobel_kernel.shape[1]]
            )
    ax[1, 0].imshow(img_correlated, cmap='binary_r')
    ax[1, 0].set_title('Correlated - by hand')

    sobel_kernel_flipped = np.flip(sobel_kernel)
    for i in range(1, img_convolved_out.shape[0] - 2):
        for j in range(1, img_convolved_out.shape[1] - 2):
            img_convolved_out[i, j] = np.sum(
                sobel_kernel_flipped * padded_img[i:i+sobel_kernel_flipped.shape[0],
                j:j+sobel_kernel_flipped.shape[1]]
            )
    ax[1, 1].imshow(img_convolved_out, cmap='binary_r')
    ax[1, 1].set_title('Convoluted - by hand')

    plt.tight_layout()
    plt.show()

def ex3():
    img = skimage.data.chelsea().astype(int)
    gray_img = np.mean(img, axis=2)

    fig, ax = plt.subplots(2, 2, figsize=(10,7))
    ax[0, 0].imshow(gray_img, cmap='binary_r', vmin=0, vmax=255)
    ax[0, 0].set_title('Original - mono')

    kernel_box = np.ones((7, 7))
    kernel_box /= np.sum(kernel_box)

    pad_width = kernel_box.shape[0] // 2
    padded_img = np.pad(gray_img, pad_width=pad_width, mode='constant')

    img_blurred_out = np.zeros_like(gray_img)

    for i in range(pad_width, padded_img.shape[0] - pad_width):
        for j in range(pad_width, padded_img.shape[1] - pad_width):
            img_blurred_out[i - pad_width, j - pad_width] = np.sum(
                kernel_box * padded_img[i - pad_width:i + pad_width + 1,
                j - pad_width:j + pad_width + 1]
            )
    ax[0, 1].imshow(img_blurred_out, cmap='binary_r', vmin=0, vmax=255)
    ax[0, 1].set_title('Blurred - by hand')

    mask = gray_img - img_blurred_out
    ax[1, 0].imshow(mask, cmap='binary_r')
    ax[1, 0].set_title('Mask')

    img_sharp = gray_img + mask
    ax[1, 1].imshow(img_sharp, cmap='binary_r', vmin=0, vmax=255)
    ax[1, 1].set_title('Filtering result')

    plt.tight_layout()
    plt.show()


def main():
    ex1_and_2()
    ex3()


if __name__ == '__main__':
    main()