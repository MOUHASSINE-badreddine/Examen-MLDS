from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # Importing TSNE
import umap.umap_ as umap  
from sklearn.cluster import KMeans

def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM array
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP array
    '''
    if method == 'ACP':
        pca = PCA(n_components=p)
        red_mat = pca.fit_transform(mat)
    elif method == 'TSNE':  # Changed from AFC to TSNE
        tsne = TSNE(n_components=3)  # Using TSNE for dimensionality reduction
        red_mat = tsne.fit_transform(mat)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=p)
        red_mat = reducer.fit_transform(mat)
    else:
        raise ValueError("Please select one of the three methods: ACP, TSNE, UMAP")
    
    return red_mat

def clust(mat, k):
    '''
    Perform clustering using KMeans

    Input:
    -----
        mat : NxP array 
        k : number of clusters
    Output:
    ------
        pred : array of predicted labels
    '''
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(mat)
    pred = kmeans.labels_
    
    return pred

# Load data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# Text data transformation
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(corpus)

# Dimensionality reduction and clustering
methods = ['ACP', 'TSNE', 'UMAP']  # Changed AFC to TSNE in the methods list
for method in methods:
    red_emb = dim_red(embeddings, 20, method)
    pred = clust(red_emb, k)
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
    
