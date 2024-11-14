import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone


def normalize(img):
    img = img.astype(np.float64)
    img -= np.min(img)
    img /= np.max(img)
    return img


def ex1():
    salinasA_img = scipy.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']
    fig, ax = plt.subplots(2, 3, figsize=(12,8))
    print(salinasA_img.shape)
    
    ax[0, 0].imshow(salinasA_img[:, :,10], cmap='gray')
    ax[0, 0].set_title('10th Band')
    ax[0, 1].imshow(salinasA_img[:, :, 100], cmap='gray')
    ax[0, 1].set_title('100th Band')
    ax[0, 2].imshow(salinasA_img[:, :, 200], cmap='gray')
    ax[0, 2].set_title('200th Band')
    
    ax[1, 0].plot(salinasA_img[10, 10, :])
    ax[1, 0].set_title('Pixel (10,10)')
    ax[1, 1].plot(salinasA_img[40, 40, :])
    ax[1, 1].set_title('Pixel (40,40)')
    ax[1, 2].plot(salinasA_img[80, 80, :])
    ax[1, 2].set_title('Pixel (80,80)')

    plt.tight_layout()
    plt.show()


def ex2():
    salinasA_img = scipy.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']
    print(salinasA_img.shape)
    fig, ax = plt.subplots(1, 2, figsize=(12,8))

    red_band = salinasA_img[:, :, 4]
    green_band = salinasA_img[:, :, 12]
    blue_band = salinasA_img[:, :, 26]

    red_band = normalize(red_band)
    green_band = normalize(green_band)
    blue_band = normalize(blue_band)

    rgb_image = np.stack((red_band, green_band, blue_band), axis=-1)

    pixels = salinasA_img.reshape(-1, salinasA_img.shape[2])
    print(pixels.shape)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(pixels)
    print(pca_result.shape)

    pca_image = pca_result.reshape(salinasA_img.shape[0], salinasA_img.shape[1], 3)
    print(pca_image.shape)

    pca_image = normalize(pca_image)

    ax[0].imshow(rgb_image)
    ax[0].set_title('RGB')

    ax[1].imshow(pca_image)
    ax[1].set_title('PCA')

    plt.tight_layout()
    plt.show()


def ex3():
    salinasA_gt = scipy.io.loadmat('SalinasA_gt.mat')['salinasA_gt']
    salinasA_img = scipy.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']
    print(f'GT original shape: {salinasA_gt.shape}')
    salinasA_gt = salinasA_gt.flatten()
    print(f'GT flattened shape: {salinasA_gt.shape}')

    mask = salinasA_gt != 0
    labels = salinasA_gt[mask]

    # rgb stuff
    red_band = salinasA_img[:, :, 4].flatten()[mask]
    green_band = salinasA_img[:, :, 12].flatten()[mask]
    blue_band = salinasA_img[:, :, 26].flatten()[mask]
    
    red_band = normalize(red_band)
    green_band = normalize(green_band)
    blue_band = normalize(blue_band)
                          
    rgb_features = np.stack((red_band, green_band, blue_band), axis=-1)
    print(f'rgb features shape: {rgb_features.shape}')

    # pca stuff
    pixels = salinasA_img.reshape(-1, salinasA_img.shape[2])
    pixels = pixels[mask]
    print(f'pixels shape: {pixels.shape}')
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(pixels)
    print(f'pca features shape: {pca_features.shape}')


    # classification
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=5)

    def evaluate(features, labels):
        accuracies = []
        for train_index, test_index in rskf.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            clf = clone(classifier)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        return np.mean(accuracies), np.std(accuracies)


    rgb_mean, rgb_std = evaluate(rgb_features, labels)
    pca_mean, pca_std = evaluate(pca_features, labels)
    full_spectral_mean, full_spectral_std = evaluate(pixels, labels)

    print("\n- - - - - - - -- - - \n")
    print(f'RGB: {rgb_mean:.3f} ({rgb_std:.3f})')
    print(f'PCA: {pca_mean:.3f} ({pca_std:.3f})')
    print(f'All: {full_spectral_mean:.3f} ({full_spectral_std:.3f})')


def main():
    ex1()
    ex2()
    ex3()


if __name__ == '__main__':
    main()

