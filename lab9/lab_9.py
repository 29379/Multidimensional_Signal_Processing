import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


def generate_circle_params():
    center_x = np.random.randint(0, 99, 1)
    center_x = center_x[0]
    center_y = np.random.randint(0, 99, 1)
    center_y = center_y[0]
    radius = np.random.randint(2, 10, 1)
    radius = radius[0]
    return center_x, center_y, radius


def add_circle(img, center_x, center_y, radius):
    rr, cc = ski.draw.disk((center_x, center_y), radius, shape=img.shape)
    img[rr, cc] = 1
    return img


def generate_img(img, iterations):
    for i in range(iterations):
        center_x, center_y, radius = generate_circle_params()
        img = add_circle(img, center_x, center_y, radius)
    return img


def erode_img(img, kernel):
    eroded_img = np.zeros_like(img)

    pad = kernel.shape[0] // 2
    img = np.pad(img, pad_width=pad)

    for i in range(0+pad, img.shape[0]-pad):
        for j in range(0+pad, img.shape[1]-pad):
            mul_product = img[i-1:i+2, j-1:j+2]*kernel
            if np.sum(mul_product) == np.sum(kernel):
                eroded_img[i-pad, j-pad] = 1
    return eroded_img


def dilate_img(img, kernel):
    dilated_img = np.zeros_like(img)
    kernel_flipped = np.flip(kernel)

    pad = kernel.shape[0] // 2
    img = np.pad(img, pad_width=pad)

    for i in range(0+pad, img.shape[0]-pad):
        for j in range(0+pad, img.shape[1]-pad):
            mul_product = img[i-1:i+2, j-1:j+2]*kernel_flipped
            dilated_img[i-pad, j-pad] = np.sum(mul_product) >= 1
    return dilated_img


def hit_or_miss(img, b1, b2):
    hit_or_miss_img = np.zeros_like(img)
    pad = b1.shape[0] // 2
    img = np.pad(img, pad_width=pad)

    for i in range(0+pad, img.shape[0]-pad):
        for j in range(0+pad, img.shape[1]-pad):
            sub_img = img[i-pad:i+pad+1, j-pad:j+pad+1]
            object_search = sub_img * b1
            background_search = (1-sub_img) * b2
            
            hit_or_miss_img[i-pad, j-pad] = np.sum(object_search) == np.sum(b1) and\
                                                np.sum(background_search) == np.sum(b2)
    return hit_or_miss_img


def ex1():  
    fig, ax = plt.subplots(1, 3, figsize=(12,6))

    custom_img = np.zeros((100, 100)).astype(int)
    custom_img = generate_img(custom_img, 100)
    kernel = np.array(
        [[1, 1, 1], 
         [1, 1, 1], 
         [1, 1, 1]]
    )
    eroded_img = erode_img(custom_img, kernel)
    difference = custom_img - eroded_img

    ax[0].imshow(custom_img, cmap='binary')
    ax[0].set_title('original')
    ax[1].imshow(eroded_img, cmap='binary')
    ax[1].set_title('erosion')
    ax[2].imshow(difference, cmap='binary')
    ax[2].set_title('difference')
    plt.tight_layout()
    plt.show()


def ex2():
    fig, ax = plt.subplots(2, 2, figsize=(12,8))

    custom_img = np.zeros((100, 100)).astype(int)
    custom_img = generate_img(custom_img, 100)
    kernel = np.array(
        [[1, 0, 0], 
         [1, 0, 0], 
         [1, 1, 1]]
    )
    eroded_img = erode_img(custom_img, kernel)
    dilated_img = dilate_img(custom_img, kernel)

    opening_img = dilate_img(eroded_img, kernel)
    closing_img = erode_img(dilated_img, kernel)

    ax[0, 0].imshow(eroded_img, cmap='binary')
    ax[0, 0].set_title('erosion')
    ax[0, 1].imshow(dilated_img, cmap='binary')
    ax[0, 1].set_title('dilatation')
    ax[1, 0].imshow(opening_img, cmap='binary')
    ax[1, 0].set_title('opening')
    ax[1, 1].imshow(closing_img, cmap='binary')
    ax[1, 1].set_title('closing')
    plt.tight_layout()
    plt.show()


def ex3():
    fig, ax = plt.subplots(2, 2, figsize=(12,8))

    custom_img = np.zeros((100, 100)).astype(int)
    custom_img = generate_img(custom_img, 100)
    kernel = np.array(
        [[1, 1, 1], 
         [1, 1, 1], 
         [1, 1, 1]]
    )

    b1 = np.zeros((9, 9)).astype(int)
    rr, cc = ski.draw.disk((4, 4), 4)
    b1[rr, cc] = 1
    intermediate = dilate_img(b1, kernel)
    b2 = intermediate - b1
    hom_img = hit_or_miss(custom_img, b1, b2)

    ax[0, 0].imshow(b1, cmap='binary')
    ax[0, 0].set_title('B1')
    ax[0, 1].imshow(b2, cmap='binary')
    ax[0, 1].set_title('B2')
    ax[1, 0].imshow(custom_img, cmap='binary')
    ax[1, 0].set_title('image')
    ax[1, 1].imshow(hom_img, cmap='binary')
    ax[1, 1].set_title('hit-or-miss')
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(1299)
    ex1()
    ex2()
    ex3()


if __name__ == '__main__':
    main()