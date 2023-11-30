import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
import numpy as np

# Dimensionality reduction function
def dim_red(mat, p, method):
    if method == 'ACP':
        pca = PCA(n_components=p)
        red_mat = pca.fit_transform(mat)
    elif method == 'TSNE':
        tsne = TSNE(n_components=p)
        red_mat = tsne.fit_transform(mat)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=p)
        red_mat = reducer.fit_transform(mat)
    else:
        raise ValueError("Method not supported")
    return red_mat

# Clustering function
def clust(mat, k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(mat)
    pred = kmeans.labels_
    return pred

# Function to filter data for only three classes
def filter_three_classes(corpus, labels, class_indices):
    filtered_corpus = []
    filtered_labels = []
    for text, label in zip(corpus, labels):
        if label in class_indices:
            filtered_corpus.append(text)
            filtered_labels.append(label)
    return filtered_corpus, filtered_labels

# Load and preprocess data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data
labels = ng20.target

# Filter the dataset for three classes
class_indices = [0, 1, 2]  # Example classes
filtered_corpus, filtered_labels = filter_three_classes(corpus, labels, class_indices)

# Initialize sentence transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define the methods
methods = ['ACP', 'TSNE', 'UMAP']

# Number of iterations and initialization of score dictionaries
num_iterations = 10
nmi_scores = {method: [] for method in methods}
ari_scores = {method: [] for method in methods}

# Iterative process
for iteration in range(num_iterations):
    # Randomly select 2000 samples
    selected_data, selected_labels = shuffle(filtered_corpus, filtered_labels, n_samples=2000)

    # Generate embeddings
    embeddings = model.encode(selected_data)

    # Iterate over each method
    for method in methods:
        # Dimensionality reduction
        red_emb = dim_red(embeddings, 2, method)  # Using 2 components

        # Clustering
        pred = clust(red_emb, len(class_indices))

        # Evaluate clustering results
        nmi_score = normalized_mutual_info_score(selected_labels, pred)
        ari_score = adjusted_rand_score(selected_labels, pred)

        # Store the scores
        nmi_scores[method].append(nmi_score)
        ari_scores[method].append(ari_score)

    print(f'Iteration {iteration+1} completed')

# Plotting
plt.figure(figsize=(12, 10))

# Plot NMI Scores
for idx, method in enumerate(methods, 1):
    plt.subplot(2, len(methods), idx)
    plt.plot(range(1, num_iterations + 1), nmi_scores[method], marker='o', label=f'NMI-{method}')
    plt.title(f'NMI Scores (Method: {method})')
    plt.xlabel('Iteration')
    plt.ylabel('NMI Score')
    plt.xticks(range(1, num_iterations + 1))
    plt.legend()

# Plot ARI Scores
for idx, method in enumerate(methods, 1):
    plt.subplot(2, len(methods), len(methods) + idx)
    plt.plot(range(1, num_iterations + 1), ari_scores[method], marker='o', label=f'ARI-{method}')
    plt.title(f'ARI Scores (Method: {method})')
    plt.xlabel('Iteration')
    plt.ylabel('ARI Score')
    plt.xticks(range(1, num_iterations + 1))
    plt.legend()

plt.tight_layout()
plt.show()
