import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


def plot_rgb_histogram(img, ax):
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    u_red, c_red = np.unique(red_channel, return_counts=True)
    u_green, c_green = np.unique(green_channel, return_counts=True)
    u_blue, c_blue = np.unique(blue_channel, return_counts=True)

    total_pixels = img.shape[0] * img.shape[1]
    probs_red = c_red / total_pixels
    probs_green = c_green / total_pixels
    probs_blue = c_blue / total_pixels

    ax.scatter(u_red, probs_red, color='red', s=5)
    ax.scatter(u_green, probs_green, color='green', s=5)
    ax.scatter(u_blue, probs_blue, color='blue', s=5)
    ax.grid(ls=':')


def ex_1_and_2():
    fig, ax = plt.subplots(6, 3, figsize=(12,15))
    img = ski.data.chelsea()
    D = 8
    L = 2 ** D
    x_values = np.linspace(0, L-1, L)
    
    identity_LUT = (np.linspace(0, L-1, L)).astype(int)
    negation_LUT = np.linspace(L-1, 0, L).astype(int)
    threshold_LUT = np.where((identity_LUT >= 50) & (identity_LUT <= 150), L-1, 0).astype(int)
    sine_LUT = ((L-1) * (np.sin(np.linspace(0, 2 * np.pi, L)) + 1) / 2).astype(int)
    
    gamma_0_3_LUT = np.copy(identity_LUT).astype(float)
    gamma_0_3_LUT -= np.min(identity_LUT)
    gamma_0_3_LUT /= np.max(identity_LUT)
    gamma_0_3_LUT = np.pow(gamma_0_3_LUT, 0.3)
    gamma_0_3_LUT = (gamma_0_3_LUT * 255).astype(int)
    
    gamma_3_LUT = np.copy(identity_LUT).astype(float)
    gamma_3_LUT -= np.min(identity_LUT)
    gamma_3_LUT /= np.max(identity_LUT)
    gamma_3_LUT = np.pow(gamma_3_LUT, 3)
    gamma_3_LUT = (gamma_3_LUT * 255).astype(int)

    ax[0, 0].scatter(x_values, identity_LUT, s=5)
    ax[1, 0].scatter(x_values, negation_LUT, s=5)
    ax[2, 0].scatter(x_values, threshold_LUT, s=5)
    ax[3, 0].scatter(x_values, sine_LUT, s=5)
    ax[4, 0].scatter(x_values, gamma_0_3_LUT, s=5)
    ax[5, 0].scatter(x_values, gamma_3_LUT, s=5)

    img_identity = identity_LUT[img]
    img_negation = negation_LUT[img]
    img_threshold = threshold_LUT[img]
    img_sine = sine_LUT[img]
    img_gamma_0_3 = gamma_0_3_LUT[img]
    img_gamma_3 = gamma_3_LUT[img]

    ax[0,1].imshow(img_identity, cmap='binary_r')
    ax[1,1].imshow(img_negation, cmap='binary_r')
    ax[2,1].imshow(img_threshold, cmap='binary_r')
    ax[3,1].imshow(img_sine, cmap='binary_r')
    ax[4,1].imshow(img_gamma_0_3, cmap='binary_r')
    ax[5,1].imshow(img_gamma_3, cmap='binary_r')

    plot_rgb_histogram(img_identity, ax[0, 2])
    plot_rgb_histogram(img_negation, ax[1, 2])
    plot_rgb_histogram(img_threshold, ax[2, 2])
    plot_rgb_histogram(img_sine, ax[3, 2])
    plot_rgb_histogram(img_gamma_0_3, ax[4, 2])
    plot_rgb_histogram(img_gamma_3, ax[5, 2])

    plt.tight_layout()
    plt.show()


def ex3():
    fig, ax = plt.subplots(2, 3, figsize=(12,9))
    img = ski.data.moon()
    ax[0, 0].imshow(img, cmap='binary_r')

    u, c = np.unique(img, return_counts=True)
    total_pixels = img.shape[0] * img.shape[1]
    probs = c / total_pixels
    ax[0,1].bar(u, probs, color='black')
    ax[0,1].grid(ls=':')

    cdf = np.cumsum(probs)
    ax[0, 2].scatter(u, cdf, color='black', s=5)
    ax[0, 2].grid(ls=':')
    
    # lut = (cdf * 255).astype(int)
    lut = np.zeros(256, dtype=int)
    lut[u] = (cdf * 255).astype(int)
    ax[1, 0].scatter(u, lut[u], color='black', s=5)
    ax[1, 0].grid(ls=':')

    img_cumsum_lut = lut[img]
    ax[1, 1].imshow(img_cumsum_lut, cmap='binary_r')

    u_cumsum_lut, c_cumsum_lut = np.unique(img_cumsum_lut, return_counts=True)
    probs_cumsum_lut = c_cumsum_lut / total_pixels
    ax[1,2].bar(u_cumsum_lut, probs_cumsum_lut, color='black')
    ax[1,2].grid(ls=':')

    plt.tight_layout()
    plt.show()


def main():
    ex_1_and_2()
    ex3()


if __name__ == '__main__':
    main()
