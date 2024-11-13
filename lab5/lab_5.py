import matplotlib.pyplot as plt
import numpy as np
import skimage


def normalize(img):
    img -= np.min(img)
    img /= np.max(img)
    return img


def ex1():
    fig, ax = plt.subplots(2, 3, figsize=(12,8))

    img_empty = np.zeros((1000, 1000)).astype(int)
    img_empty[500:520, 460:550] = 1

    # fft_img = np.fft.ifftshift(img_empty)
    fft_img = np.fft.fft2(img_empty)    
    fft_img_shifted = np.fft.fftshift(fft_img)  

    fft_transformed_real = np.log(np.abs(np.real(fft_img_shifted)) + 1)
    fft_transformed_imag = np.log(np.abs(np.imag(fft_img_shifted)) + 1)

    phase_shift = np.arctan2(np.imag(fft_img_shifted), np.real(fft_img_shifted))

    magnitude = np.log(np.abs(fft_img_shifted) + 1)
    inverse_fft = np.fft.ifft2(np.fft.ifftshift(fft_img_shifted))

    #   -------------------------------------

    ax[0, 0].imshow(img_empty, cmap='magma')
    ax[0, 0].set_title('Original')

    ax[0, 1].imshow(fft_transformed_real, cmap='magma')
    ax[0, 1].set_title('Real part')
    
    ax[0, 2].imshow(fft_transformed_imag, cmap='magma')
    ax[0, 2].set_title('Imaginary part')

    ax[1, 0].imshow(phase_shift, cmap='magma')    
    ax[1, 0].set_title('Phase Shift')

    ax[1, 1].imshow(magnitude, cmap='magma')
    ax[1, 1].set_title('Magnitude')

    ax[1, 2].imshow(np.real(inverse_fft), cmap='magma')
    ax[1, 2].set_title('Inverse FFT')
    plt.tight_layout()
    plt.show()


def ex2():
    fig, ax = plt.subplots(2, 3, figsize=(12,8))

    x = np.linspace(0, 15, 1000)
    y = np.linspace(0, 15, 1000)
    X, Y = np.meshgrid(x, y)

    img_zeros = np.zeros((1000, 1000))
    amplitudes = np.random.uniform(0, 10, 5)
    angles = np.random.uniform(0, 5 * np.pi, 5)
    wavelengths = np.random.uniform(1, 10, 5)

    for values in zip(amplitudes, angles, wavelengths):
        amp, ang, wl = values
        # for x in range (0, img_zeros.shape[0]):
        #     for y in range (0, img_zeros.shape[1]):
        #         img_zeros[x, y] += amp * np.sin(x * np.cos(ang) + y * np.sin(ang) / wl)
        img_zeros += amp * np.sin(X * np.cos(ang) + Y * np.sin(ang) / wl)

    img_zeros = normalize(img_zeros)

    fft_img = np.fft.fft2(img_zeros)
    fft_img_shifted = np.fft.fftshift(fft_img)

    fft_transformed_real = np.log(np.abs(np.real(fft_img_shifted)) + 1)
    fft_transformed_imag = np.log(np.abs(np.imag(fft_img_shifted)) + 1)

    phase_shift = np.arctan2(np.imag(fft_img_shifted), np.real(fft_img_shifted))

    magnitude = np.log(np.abs(fft_img_shifted) + 1)
    inverse_fft = np.fft.ifft2(np.fft.ifftshift(fft_img_shifted))

    ax[0, 0].imshow(img_zeros, cmap='magma')
    ax[0, 0].set_title('Original')

    ax[0, 1].imshow(fft_transformed_real, cmap='magma')
    ax[0, 1].set_title('Real part')

    ax[0, 2].imshow(fft_transformed_imag, cmap='magma')
    ax[0, 2].set_title('Imaginary part')

    ax[1, 0].imshow(phase_shift, cmap='magma')
    ax[1, 0].set_title('Phase Shift')

    ax[1, 1].imshow(magnitude, cmap='magma')
    ax[1, 1].set_title('Magnitude')

    ax[1, 2].imshow(np.real(inverse_fft), cmap='magma')
    ax[1, 2].set_title('Inverse FFT')

    plt.tight_layout()
    plt.show()


def ex3():
    fig, ax = plt.subplots(2, 3, figsize=(12,8))
    img = skimage.data.chelsea().astype(int)
    gray_img = np.mean(img, axis=2)

    fft_img = np.fft.fft2(gray_img)
    fft_img_shifted = np.fft.fftshift(fft_img)

    fft_transformed_real = np.log(np.abs(np.real(fft_img_shifted)) + 1)
    fft_transformed_imag = np.log(np.abs(np.imag(fft_img_shifted)) + 1)

    real_part = np.real(fft_img_shifted)
    imag_part = np.imag(fft_img_shifted) * 1j

    inverse_real = np.fft.ifft2(np.fft.ifftshift(real_part))
    inverse_imag = np.fft.ifft2(np.fft.ifftshift(imag_part))
    inverse_complex = np.fft.ifft2(np.fft.ifftshift(fft_img_shifted))

    red_channel = inverse_real.real
    green_channel = inverse_imag.real
    blue_channel = inverse_complex.real
    red_channel = normalize(red_channel)
    green_channel = normalize(green_channel)
    blue_channel = normalize(blue_channel)

    magnitude = np.log(np.abs(fft_img_shifted) + 1)
    color_img = np.stack((red_channel, green_channel, blue_channel), axis=-1)

    ax[0, 0].imshow(gray_img, cmap='gray')
    ax[0, 0].set_title('Original')

    ax[0, 1].imshow(magnitude)
    ax[0, 1].set_title('Magnitude')

    ax[0, 2].imshow(color_img, cmap='gray')
    ax[0, 2].set_title('Colorful image')

    ax[1, 0].imshow(red_channel, cmap='Reds_r')
    ax[1, 0].set_title('Red channel')

    ax[1, 1].imshow(green_channel, cmap='Greens_r')
    ax[1, 1].set_title('Green channel')

    ax[1, 2].imshow(blue_channel, cmap='Blues_r')
    ax[1, 2].set_title('Blue channel')

    plt.tight_layout()
    plt.show()


def main():
    ex1()
    ex2()
    ex3()


if __name__ == '__main__':
    main()
    