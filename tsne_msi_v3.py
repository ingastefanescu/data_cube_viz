import os
import numpy as np
from skimage import io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = io.imread(image_path, as_gray=True)  # Read images in grayscale
        images.append(image.flatten())  # Flatten the image and append to the list

    return np.array(images)

def apply_tsne_to_images(images, no_dims=3, perplexity=30.0):
    n_images, n_pixels = images.shape
    tsne_results = np.zeros((n_images, no_dims))

    for i in range(n_images):
        image = images[i].reshape(1, -1)  # Reshape to (1, n_pixels) for PCA and t-SNE
        n_components = min(n_pixels, 50)

        if n_components < n_pixels:
            # Apply PCA for dimensionality reduction before t-SNE
            image_pca = PCA(n_components=n_components).fit_transform(image)
            # Apply t-SNE
            tsne = TSNE(n_components=no_dims, perplexity=perplexity)
            tsne_result = tsne.fit_transform(image_pca)
            tsne_results[i, :] = tsne_result.flatten()  # Flatten the result to a 1D vector
        else:
            # If the number of pixels is less than the desired components, skip PCA
            tsne = TSNE(n_components=no_dims, perplexity=perplexity)
            tsne_result = tsne.fit_transform(image)
            tsne_results[i, :] = tsne_result.flatten()

    return tsne_results

# Specify the folder containing your images
folder_path = "/msi"

# Read images from the folder
msi_data = read_images_from_folder(folder_path)

# Apply t-SNE to each image
tsne_results = apply_tsne_to_images(msi_data, no_dims=3, perplexity=30.0)

# tsne_results is a 2D array where tsne_results[i, :] is the 3D t-SNE result for the i-th image
# You can access individual results like tsne_results[0, :] for the first image
