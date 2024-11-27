import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


def normalize(img):
    img = img.astype(np.float64)
    img -= np.min(img)
    img /= np.max(img)
    return img


def generate_degen_fun(size):
    degen_matrix = np.zeros((size,size)).astype(int)
    for i in range(size):
        for j in range(size):
            if i != j:
                degen_matrix[i, j] = -1
            else:
                degen_matrix[i, j] = 9
    return (1/(-39))*degen_matrix


def wiener_filter(G, H, lam):
    return (G/H)*(1/(1+(lam/(H*H))))


def ex1():
    img = ski.data.chelsea()
    img = np.mean(img, axis=2)
    img = np.resize(img, new_shape=(img.shape[0]+1, 451))
    fig, ax = plt.subplots(3, 2, figsize=(10,7))

    fft_img = np.fft.fft2(img)
    fft_img_shifted = np.fft.fftshift(fft_img)
    magnitude = np.log(np.abs(fft_img_shifted) + 1)

    sobel_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    # sobel_kernel = np.array([
    #     [0, 1, 2],
    #     [-1, 0, 1],
    #     [-2, -1, 0]
    # ])  

    pad_height = img.shape[0] - sobel_kernel.shape[0]
    pad_width = img.shape[1] - sobel_kernel.shape[1]
    sobel_kernel = np.pad(sobel_kernel, ((0, pad_height), (0, pad_width)))

    fft_sobel_kernel = np.fft.fft2(sobel_kernel, s=img.shape)
    fft_sobel_kernel_shifted = np.fft.fftshift(fft_sobel_kernel)
    magnitude_fft_sobel = np.log(np.abs(fft_sobel_kernel_shifted) + 1)

    frequency_filtration = fft_img_shifted * fft_sobel_kernel_shifted
    filtered_transform = np.log(np.abs(np.real(frequency_filtration)) + 1)

    filtered_img = np.fft.ifftshift(frequency_filtration)
    inverse_transform_img = np.fft.ifft2(filtered_img)
    inverse_transform_img = np.real(inverse_transform_img)

    ax[0,0].imshow(img, cmap='gray')
    ax[0,0].set_title('Original')

    ax[0,1].imshow(magnitude, cmap='magma')
    ax[0,1].set_title('Fourier transform')

    ax[1,0].imshow(sobel_kernel, cmap='gray')
    ax[1,0].set_title('Sobel kernel')

    ax[1, 1].imshow(magnitude_fft_sobel, cmap='magma')
    ax[1, 1].set_title('Fourier transform Sobel')

    ax[2, 0].imshow(filtered_transform, cmap='magma')
    ax[2, 0].set_title('Fourier transform filtered')

    ax[2, 1].imshow(inverse_transform_img, cmap='gray')
    ax[2, 1].set_title('Inverse transform')

    plt.tight_layout()
    plt.show()

def ex2():
    img = plt.imread('judasz_filtered.png')
    img_mono = img[:, :, 1]
    fig, ax = plt.subplots(3, 2, figsize=(10,7))

    degen_matrix = generate_degen_fun(13)
    # print(degen_matrix)
    pad_height = img_mono.shape[0] - degen_matrix.shape[0]
    pad_width = img_mono.shape[1] - degen_matrix.shape[1]
    degen_matrix = np.pad(degen_matrix, ((0, pad_height), (0, pad_width)))

    fft_img = np.fft.fft2(img_mono)
    fft_img_shifted = np.fft.fftshift(fft_img)
    fourier_transform_img = np.log(np.abs(fft_img_shifted) + 1)

    fft_degen_matrix = np.fft.fft2(degen_matrix, s=img_mono.shape)
    fft_degen_matrix_shifted = np.fft.fftshift(fft_degen_matrix)
    fourier_transform_degen = np.log(np.abs(fft_degen_matrix_shifted) + 1)

    deconvolution = fft_img_shifted / fft_degen_matrix_shifted
    deconvolution_transform = np.log(np.abs(np.real(deconvolution)) + 1)

    filtered_img = np.fft.ifftshift(deconvolution)
    inverse_transform_img = np.fft.ifft2(filtered_img)
    inverse_transform_img = np.real(inverse_transform_img)

    ax[0,0].imshow(img_mono, cmap='gray')
    ax[0,0].set_title("Judasz (degenerated)")

    ax[0, 1].imshow(fourier_transform_img, cmap='magma')
    ax[0, 1].set_title('Fourier transform')

    ax[1, 0].imshow(degen_matrix, cmap='gray')
    ax[1, 0].set_title('Degen matrix')

    ax[1, 1].imshow(fourier_transform_degen, cmap='magma')
    ax[1, 1].set_title('Fourier transform on degen matrix')
    
    ax[2, 0].imshow(deconvolution_transform, cmap='magma')
    ax[2, 0].set_title('Deconvoluted FT')

    ax[2, 1].imshow(inverse_transform_img, cmap='gray')
    ax[2, 1].set_title('Deconvoluted image')

    plt.tight_layout()
    plt.show()


def ex3():
    img = plt.imread('judasz_filtered.png')
    img_mono = img[:, :, 1]
    fig, ax = plt.subplots(3, 2, figsize=(10,7))

    degen_matrix = generate_degen_fun(13)
    # print(degen_matrix)
    pad_height = img_mono.shape[0] - degen_matrix.shape[0]
    pad_width = img_mono.shape[1] - degen_matrix.shape[1]
    degen_matrix = np.pad(degen_matrix, ((0, pad_height), (0, pad_width)))

    fft_img = np.fft.fft2(img_mono)
    fft_img_shifted = np.fft.fftshift(fft_img)
    fourier_transform_img = np.log(np.abs(fft_img_shifted) + 1)

    fft_degen_matrix = np.fft.fft2(degen_matrix, s=img_mono.shape)
    fft_degen_matrix_shifted = np.fft.fftshift(fft_degen_matrix)
    fourier_transform_degen = np.log(np.abs(fft_degen_matrix_shifted) + 1)

    deconvolution_transform_wiener = wiener_filter(fft_img_shifted, fft_degen_matrix_shifted, 0.02)

    filtered_img = np.fft.ifftshift(deconvolution_transform_wiener)
    inverse_transform_img = np.fft.ifft2(filtered_img)
    inverse_transform_img = np.real(inverse_transform_img)

    ax[0,0].imshow(img_mono, cmap='gray')
    ax[0,0].set_title("Judasz (degenerated)")

    ax[0, 1].imshow(fourier_transform_img, cmap='magma')
    ax[0, 1].set_title('Fourier transform')

    ax[1, 0].imshow(degen_matrix, cmap='gray')
    ax[1, 0].set_title('h')

    ax[1, 1].imshow(fourier_transform_degen, cmap='magma')
    ax[1, 1].set_title('H')
    
    ax[2, 0].imshow(np.log(np.abs(deconvolution_transform_wiener) + 1), cmap='magma')
    ax[2, 0].set_title('Deconvoluted FT')

    ax[2, 1].imshow(inverse_transform_img, cmap='gray')
    ax[2, 1].set_title('Judasz (deconvoluted)')

    plt.tight_layout()
    plt.show()


def main():
    ex1()
    ex2()
    ex3()


if __name__ == '__main__':
    main()