import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy.ndimage import gaussian_filter, correlate, median_filter


def normalize(img):
    img = img.astype(np.float64)
    img -= np.min(img)
    img /= np.max(img)
    return img


def edge_detection_Laplacian(img):
    L = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    img_correlated = correlate(img, L)
    img_correlated[img_correlated < 0] = 0
    return img_correlated


def edge_detection_gradient_magnitude(img):
    Pz = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    Py = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    Gx = correlate(img, Pz)
    Gy = correlate(img, Py)
    
    Gx[Gx < 0] = 0
    Gy[Gy < 0] = 0
    
    M = np.abs(Gx) + np.abs(Gy)
    return M


def plot_histogram_log(img, ax):
    ax.hist(img.flatten(), bins=256)
    ax.set_yscale('log')
    ax.grid(ls=':')


def otsu_thresholding(img, num_pixels_total):
    variances = []
    for threshold in range(256):
        img_edge_gradient_thresholding = (img < threshold)

        pixels_1 = img[img_edge_gradient_thresholding]
        pixels_0 = img[~img_edge_gradient_thresholding]

        Pk_1 = np.abs(len(pixels_1)) / num_pixels_total
        mk_1 = np.mean(pixels_1)
        mg_1 = np.mean(img)

        Pk_0 = np.abs(len(pixels_0)) / num_pixels_total
        mk_0 = np.mean(pixels_0)
        mg_0 = np.mean(img)

        v = Pk_1 * (mk_1 - mg_1)**2 + Pk_0 * (mk_0 - mg_0)**2
        if np.isnan(v):
            v = 0
        variances.append(v)
    
    best_threshold = np.argmax(variances)
    return (variances, best_threshold)


def ex1():
    fig, ax = plt.subplots(1, 3, figsize=(15,8))

    img = ski.data.chelsea()
    img = np.mean(img, axis=2)

    img = gaussian_filter(img, sigma=1)
    img_edge_laplacian = edge_detection_Laplacian(img.copy())
    img_edge_gradient = edge_detection_gradient_magnitude(img.copy())
    
    ax[0].imshow(img, cmap='binary_r')
    ax[0].set_title('Original')
    ax[1].imshow(img_edge_laplacian, cmap='binary')
    ax[1].set_title('Edge detection - Laplacian')
    ax[2].imshow(img_edge_gradient, cmap='binary')
    ax[2].set_title('Edge detection - gradient magnitude')
    plt.tight_layout()
    plt.show()

    return (img, img_edge_laplacian, img_edge_gradient)


def ex2(img, img_edge_laplacian, img_edge_gradient):
    fig, ax = plt.subplots(2, 4, figsize=(15,8))

    img_edge_laplacian = normalize(img_edge_laplacian)
    img_edge_gradient = normalize(img_edge_gradient)

    # thresholding
    img_edge_laplacian_thresholding = (img_edge_laplacian > 0.1)
    img_edge_gradient_thresholding = (img_edge_gradient > 0.1) 

    #   median filter
    img_edge_laplacian_median_filter = median_filter(img_edge_laplacian, size=21)
    img_edge_gradient_median_filter = median_filter(img_edge_gradient, size=21)

    #   adaptive thresholding
    img_edge_laplacian_adaptive_thresholding = (img_edge_laplacian > median_filter(img_edge_laplacian, size=21))
    img_edge_gradient_adaptive_thresholding = (img_edge_gradient > median_filter(img_edge_gradient, size=21))

    ax[0,0].imshow(img_edge_laplacian, cmap='binary')
    ax[0,0].set_title('Edge detection - Laplacian')
    ax[1,0].imshow(img_edge_gradient, cmap='binary')
    ax[1,0].set_title('Edge detection - gradient magnitude')

    ax[0,1].imshow(img_edge_laplacian_thresholding, cmap='binary')
    ax[0,1].set_title('Laplacian - thresholding')
    ax[1,1].imshow(img_edge_gradient_thresholding, cmap='binary')
    ax[1,1].set_title('Gradient magnitude - thresholding')

    ax[0,2].imshow(img_edge_laplacian_median_filter, cmap='binary')
    ax[0,2].set_title('Laplacian - median filter')
    ax[1,2].imshow(img_edge_gradient_median_filter, cmap='binary')
    ax[1,2].set_title('Gradient magnitude - median filter')

    ax[0,3].imshow(img_edge_laplacian_adaptive_thresholding, cmap='binary')
    ax[0,3].set_title('Laplacian - adaptive thresholding')
    ax[1,3].imshow(img_edge_gradient_adaptive_thresholding, cmap='binary')
    ax[1,3].set_title('Gradient magnitude - adaptive thresholding')

    plt.tight_layout()
    plt.show()


def ex3(img_edge_laplacian):
    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    
    img_edge_laplacian = normalize(img_edge_laplacian)
    img_edge_laplacian_8bit = (img_edge_laplacian * 255).astype(np.uint8)

    num_pixels_total = img_edge_laplacian.shape[0] * img_edge_laplacian.shape[1]
    (variances, best_threshold) = otsu_thresholding(img_edge_laplacian_8bit.copy(), num_pixels_total)
    img_edge_laplacian_8bit_otsu_thresholded = (img_edge_laplacian_8bit > best_threshold)

    ax[0,0].imshow(img_edge_laplacian_8bit, cmap='binary')
    ax[0,0].set_title('Laplacian - 8-bit')

    plot_histogram_log(img_edge_laplacian_8bit, ax[0,1])

    ax[1,0].plot(variances)
    ax[1,0].vlines(best_threshold, 0, max(variances), color='red', linestyles='dotted')
    ax[1,0].set_title('Variances')

    ax[1,1].imshow(img_edge_laplacian_8bit_otsu_thresholded, cmap='binary')
    ax[1,1].set_title('Otsu thresholding')

    plt.tight_layout()
    plt.show()


def main():
    (img, img_edge_laplacian, img_edge_gradient) = ex1()
    ex2(img, img_edge_laplacian, img_edge_gradient)
    ex3(img_edge_laplacian)


if __name__ == '__main__':
    main()