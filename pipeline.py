from img2vec_pytorch import Img2Vec
from PIL import Image
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shutil


def get_embeddings(list_of_PIL_images):
    img2vec = Img2Vec(cuda=False)
    vectors = img2vec.get_vec(list_of_PIL_images)
    return vectors


def get_pca(vectors):
    pca = PCA(n_components=32)
    result = pca.fit_transform(vectors)
    return result


def get_kmeans(pca):
    kmeans = KMeans(n_clusters=10)
    clusters = kmeans.fit_predict(pca)
    return clusters


def main():

    image_paths = os.listdir("./images")
    list_of_PIL_images = [
        Image.open(os.path.join("images", path)).convert("RGB") for path in image_paths
    ]

    vectors = get_embeddings(list_of_PIL_images)
    pca = get_pca(vectors)
    clusters = get_kmeans(pca)

    for i, image_path in enumerate(image_paths):
        print(image_path, clusters[i])
        shutil.copyfile(
            os.path.join("images", image_path),
            os.path.join("images_out", str(clusters[i]) + "_" + image_path),
        )


if __name__ == "__main__":
    main()
    print("Done!")
