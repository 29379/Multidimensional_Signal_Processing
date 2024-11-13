import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy.spatial.distance import cdist

#   ZAD 1 - - - - - - - - - - - - -

def ex1():
    fig, ax = plt.subplots(2, 2, figsize=(10,7))
    img = ski.data.chelsea()

    ax[0,0].imshow(img, cmap='gray')
    ax[0, 0].set_title('Original')

    # axes = tuple(range(img.ndim))
    gray_img = np.mean(img, axis=2)

    # ax[0,1].imshow(gray_img, cmap='binary_r')
    # ax[0, 1].set_title('Grayscale')

    smaller_img = gray_img[::8, ::8]
    ax[0,1].imshow(smaller_img, cmap='binary_r')
    ax[0, 1].set_title('8x smaller')

    angle = np.pi / 12
    rotation_tmatrix = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ]
    tform_rotation = ski.transform.AffineTransform(matrix=rotation_tmatrix)
    rotated_img = ski.transform.warp(smaller_img, inverse_map=tform_rotation.inverse)
    ax[1,0].imshow(rotated_img, cmap='binary_r')
    ax[1,0].set_title('15 degrees rotation')

    shear_matrix = [
        [1, 0.5, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    tform_shear = ski.transform.AffineTransform(matrix=shear_matrix)
    sheared_img = ski.transform.warp(smaller_img, inverse_map=tform_shear.inverse)
    ax[1,1].imshow(sheared_img, cmap='binary_r')
    ax[1,1].set_title('0.5 shear x-axis')

    plt.tight_layout()
    plt.show()
    return smaller_img

#   - - - - - - - - -- - - - - - - -
#   ZAD 2 - - - - - - - - - - - - -

def ex2(img):
    fig, ax = plt.subplots(1, 2, figsize=(9,6), sharex=True, sharey=True)
    x, y = img.shape
    # coords = np.meshgrid(
    #     x, y
    # )
    # print(f'{x} + {y}')
    # coords = np.meshgrid(
    #     np.arange(x),
    #     np.arange(y)
    # )
    # coords = np.meshgrid(
    #     range(x),
    #     range(y)
    # )
    coords = []
    for i in range(x):
        for j in range(y):
            coords.append((i, j))
    coords = np.array(coords)
    ax[0].set_title('Original')
    ax[0].scatter(coords[:,0], coords[:,1], c=img, cmap='binary_r')


    extended_coords = []
    for i in range(coords.shape[0]):
        c, r = coords[i]
        extended_coords.append((c, r, 1))
    extended_coords = np.array(extended_coords)

    angle = np.pi / 12
    rotation_tmatrix = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ]
    new_coords = extended_coords@rotation_tmatrix
    ax[1].set_title('Rotated 15 degrees')
    ax[1].scatter(new_coords[:,0], new_coords[:,1], c=img, cmap='binary_r')

    plt.tight_layout()
    plt.show()
    return coords

#   - - - - - - - - -- - - - - - - -
#   ZAD 3 - - - - - - - - - - - - -

def ex3(img, coords):
    fig, ax = plt.subplots(1, 3, figsize=(10,7), sharex=True, sharey=True)

    num_points = 1000
    random_indices = np.random.choice(coords.shape[0], num_points, replace=False)
    random_indices.sort()

    lost_signal = []
    intensities = []
    for i in random_indices:
        lost_signal.append(coords[i])
        x = coords[i][0]
        y = coords[i][1]
        intensities.append(img[x][y])
    lost_signal = np.array(lost_signal)
    intensities = np.array(intensities)

    ax[0].set_title('Original')
    ax[0].scatter(coords[:,0], coords[:,1], c=img, cmap='binary_r')
    ax[1].set_title('Lost signal')
    ax[1].scatter(lost_signal[:,0], lost_signal[:,1], c=intensities, cmap='binary_r')

    # Nearest neighbor interpolation
    new_height, new_width = 300, 400
    x_coords = np.linspace(0, img.shape[0] - 1, new_height)
    y_coords = np.linspace(0, img.shape[1] - 1, new_width)
    new_coords = np.array([(x, y) for x in x_coords for y in y_coords])

    distances = cdist(new_coords, lost_signal)
    nearest_indices = np.argmin(distances, axis=1)
    # interpolated_img = np.array([intensities[i] for i in nearest_indices]).reshape(new_height, new_width)
    interpolated_img = intensities[nearest_indices].reshape(new_height, new_width)

    ax[2].set_title('Interpolated Image')
    ax[2].scatter(new_coords[:,0], new_coords[:,1], c=interpolated_img.flatten(), cmap='binary_r')

    plt.tight_layout()
    plt.show()

#   - - - - - - - - -- - - - - - - -

def main():
    img = ex1()
    coords = ex2(img)
    ex3(img, coords)

if __name__ == "__main__":
    main()

#   - - - - - - - - -- - - - - - - -




