import matplotlib.pyplot as plt
import skimage as ski
import numpy as np


def normalize(img):
    img = img.astype(np.float64)    
    for i in range(9):
        img[:,i] -= np.min(img[:,i])
        img[:,i] /= np.max(img[:,i])
    return img


def horizontal_gradient(img):
    Gx = np.empty((128, 128))
    for x in range(128):
        for y in range(128):
            if x == 0 or x == 127:
                Gx[x, y] = 0
            else:
                Gx[x, y] = img[x + 1, y] - img[x - 1, y]
    return Gx

def vertical_gradient(img):
    Gy = np.empty((128, 128))
    for x in range(128):
        for y in range(128):
            if y == 0 or y == 127:
                Gy[x, y] = 0
            else:
                Gy[x, y] = img[x, y + 1] - img[x, y - 1]
    return Gy


def ex1():
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    img = ski.data.camera()
    img = ski.transform.resize(img, (128, 128))

    Gx = horizontal_gradient(img)
    Gy = vertical_gradient(img)

    mag = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan(Gy / Gx)
    # print(np.min(angle), np.max(angle))

    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(Gx, cmap='gray')
    ax[0, 1].set_title('Gx')
    ax[0, 2].imshow(Gy, cmap='gray')
    ax[0, 2].set_title('Gy')
    ax[1, 0].imshow(mag, cmap='gray') 
    ax[1, 0].set_title('Magnitude')
    ax[1, 1].imshow(angle, cmap='gray')
    ax[1, 1].set_title('Angle')

    plt.tight_layout()
    plt.show()

    return img, Gx, Gy, mag, angle
    

def ex2(img, Gx, Gy, mag, angle):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    s = 8               # size of the aggregating cell window
    bins = 9            # number of bins in the histogram
    step = np.pi / bins # width of the bin

    mask = np.zeros_like(img)
    cell_id = 0
    for i in range(0, img.shape[0], s):
        for j in range(0, img.shape[1], s):
            mask[i:i+s, j:j+s] = cell_id
            cell_id += 1
   
    num_cells = cell_id
    hog = np.zeros((num_cells, bins))

    for cell in range(num_cells):
        ang_v = angle[mask == cell]
        mag_v = mag[mask == cell]

        for bin in range(bins):
            start = (bin * step) - (np.pi / 2)
            end = start + step
            b_mask = (ang_v >= start) & (ang_v < end)
            hog[cell, bin] = np.sum(mag_v[b_mask])

    hog = normalize(hog)
    hog_r = hog.reshape((16, 16, 9))   
    hog1 = hog_r[:, :, :3]
    hog2 = hog_r[:, :, 3:6]
    hog3 = hog_r[:, :, 6:]

    ax[0, 0].imshow(mask, cmap='seismic')
    ax[0, 0].set_title('Mask')
    ax[0, 1].imshow(hog1, cmap='seismic')
    ax[0, 1].set_title('Hog 0:3')
    ax[1, 0].imshow(hog2, cmap='seismic')
    ax[1, 0].set_title('Hog 3:6')
    ax[1, 1].imshow(hog3, cmap='seismic')
    ax[1, 1].set_title('Hog 6:9')

    plt.tight_layout()
    plt.show()

    return s, bins, step, num_cells, mask, hog


def ex3(img, s, bins, step, num_cells, mask, hog):
    fig, ax = plt.subplots(figsize=(8, 8))
    angles = np.linspace(-80, 80, bins, endpoint=False)

    img_out = np.zeros_like(img)

    for i in range(256):
        cell = np.zeros((8, 8))
        for j in range(bins):
            cell_2 = np.zeros((8, 8))
            cell_2[4] = 1
            cell_2 = ski.transform.rotate(cell_2, angles[j])
            cell_2 = cell_2 * hog[i, j]
            cell += cell_2
        img_out[mask == i] = cell.flatten()

    ax.imshow(img_out)

    plt.tight_layout()
    plt.show()


def main():
    img, Gx, Gy, mag, angle = ex1()
    s, bins, step, num_cells, mask, hog = ex2(img, Gx, Gy, mag, angle)
    ex3(img, s, bins, step, num_cells, mask, hog)


if __name__ == '__main__':
    main()
    