import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


def ex1():
    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ground_truth = np.zeros((100, 100), dtype=np.uint8)
    for i in range(3):
        radius = np.random.randint(10, 40)
        center_x = np.random.randint(radius, 100 - radius)
        center_y = np.random.randint(radius, 100 - radius)
        rr, cc = ski.draw.disk((center_x, center_y), radius, shape=img.shape)

        pixel_value = np.random.randint(100, 255)
        channel = np.random.randint(0, 3)
        img[rr, cc, channel] += pixel_value
        ground_truth[rr, cc] += pixel_value

    ground_truth = ski.measure.label(ground_truth, background=0)

    noise = np.random.normal(0, 16, img.shape)
    img = img + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    ground_truth = np.clip(ground_truth, 0, 255).astype(np.uint8)

    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[1].imshow(ground_truth)
    ax[1].set_title('Ground truth')

    plt.tight_layout()
    plt.show()

    return img, ground_truth


def ex2(img, ground_truth):
    ground_truth_reshaped = ground_truth.ravel()
    x_coords, y_coords = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    values = []
    for x, y in coords:
        values.append(img[y, x])
    values = np.array(values)

    X = np.hstack((coords, values))

    scaler = StandardScaler()
    X_normalised = scaler.fit_transform(X)
    y = ground_truth_reshaped
    print(f"X shape: {X_normalised.shape}")
    print(f"y shape: {y.shape}")
    print(f"First object from X: {X_normalised[0]}")
    print(f"First label from y: {y[0]}")

    return X_normalised, y
    

def ex3(X, y, img, ground_truth):
    fig, ax = plt.subplots(2, 3, figsize=(12, 12))
    ax = ax.flatten()

    models = {
        'KMeans': KMeans(),
        'MiniBatchKMeans': MiniBatchKMeans(),
        'Birch': Birch(),
        'DBSCAN': DBSCAN(),
    }

    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[1].imshow(ground_truth)
    ax[1].set_title('Ground truth')

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.fit_predict(X)
        score = adjusted_rand_score(y, y_pred)

        ax[i+2].set_title(f"{name} - score: {score:.3f}")
        clustered_img = y_pred.reshape(img.shape[0], img.shape[1])
        ax[i+2].imshow(clustered_img)

    plt.tight_layout()
    plt.show()


def main():
    img, ground_truth = ex1()
    X, y = ex2(img, ground_truth)
    ex3(X, y, img, ground_truth)


if __name__ == '__main__':
    main()
