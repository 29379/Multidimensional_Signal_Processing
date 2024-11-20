import pywt
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, correlate


def cast_list(test_list, data_type):
    return list(map(data_type, test_list))


def normalize(signal):
    signal /= np.sum(signal)
    return signal


def ex1():
    fig, ax = plt.subplots(9, 1, figsize=(12,9))
    fig.align_ylabels(ax[:])

    signal = np.load('signal.npy')

    transformed_signal = pywt.wavedec(signal, 'db7', level=7)
    
    ax[0].plot(signal, color=plt.cm.coolwarm(np.random.random()))
    # ax[0].set_title('Signal', color='black', fontsize=10, rotation='vertical', x=-0.039, y=-0.03)
    ax[0].set_ylabel('Signal')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].grid(True)

    for i in range(0, 8):
        ax[i + 1].plot(transformed_signal[i], color=plt.cm.coolwarm(np.random.random()))
        # ax[i + 1].set_title(f'Level {i+1}', color='black', fontsize=10, rotation='vertical', x=-0.039, y=-0.03)
        ax[i + 1].set_ylabel(f'Level {i+1}')
        ax[i + 1].spines['top'].set_visible(False)
        ax[i + 1].spines['right'].set_visible(False)
        ax[i + 1].grid(True)

    plt.tight_layout()
    plt.show()


def ex2():
    fig, ax = plt.subplots(3, 2, figsize=(12,9))
    img = skimage.data.chelsea().astype(int)
    gray_img = np.mean(img, axis=2)

    wavelet_arr = [0, 0, 2, 5, 5, -3, -5, -6, 0, 0]
    wavelet_casted = np.array(cast_list(wavelet_arr, np.float64))
    wavelet_norm = normalize(wavelet_casted)
    
    wavelet_2d = wavelet_norm[:, None] * wavelet_norm[None, :]
    wavelet_2d = wavelet_2d.reshape((10, 10))
    resized_wavelet = resize(wavelet_2d, (20, 20), anti_aliasing=True)
    filtered_wavelet = gaussian_filter(resized_wavelet, sigma=2)

    img_correlated = np.zeros_like(gray_img)
    img_correlated = correlate(gray_img, filtered_wavelet)

    ax[0, 0].plot(wavelet_arr)
    ax[0, 0].set_title('1D wavelet prototype')

    ax[0, 1].plot(wavelet_norm)
    ax[0, 1].set_title('1D wavelet prototype normalized')
    
    ax[1, 0].imshow(wavelet_2d)
    ax[1, 0].set_title('2D wavelet prototype')

    ax[1, 1].imshow(filtered_wavelet)
    ax[1, 1].set_title('2D wavelet')

    ax[2, 0].imshow(gray_img)

    ax[2, 1].imshow(img_correlated)

    plt.tight_layout()
    plt.show()
    return filtered_wavelet


def ex3(wavelet):
    fig, ax = plt.subplots(4, 4, figsize=(16,9))
    # ax = ax.ravel()
    img = skimage.data.chelsea().astype(int)
    gray_img = np.mean(img, axis=2)

    sizes = np.linspace(2, 32, 16).astype(int)
    sizes_bundles = np.array_split(sizes, 4)    

    for i, bundle in enumerate(sizes_bundles):
        for j, size in enumerate(bundle):
            resized_wavelet = resize(wavelet, (size, size), anti_aliasing=True)
            filtered_img = correlate(gray_img, resized_wavelet)
            ax[i, j].imshow(filtered_img)
            ax[i, j].set_title(f'Size: {size}')

    plt.tight_layout()
    plt.show()


def main():
    ex1()
    wavelet = ex2()
    ex3(wavelet)

if __name__ == "__main__":
    main()
