# Examen-MLDS

## Projet d'Analyse de Clustering de Données Textuelles

### Aperçu du Projet
Ce projet se concentre sur le clustering d'un ensemble de données textuelles (NG20) en utilisant différentes techniques de réduction de dimensionnalité, y compris UMAP (Uniform Manifold Approximation and Projection), t-SNE (t-distributed Stochastic Neighbor Embedding) et ACP (Analyse en Composantes Principales). L'objectif du projet est d'explorer les regroupements naturels au sein des données textuelles pour découvrir des modèles sous-jacents.

### Méthodologie
Trois différentes techniques de réduction de dimensionnalité ont été utilisées pour transformer des données textuelles de haute dimension en un espace de dimension inférieure. Ces techniques comprennent :

- **UMAP** : Une technique de réduction de dimensionnalité non linéaire qui est particulièrement efficace pour visualiser des clusters ou des groupes dans des données de haute dimension.
- **t-SNE** : Une autre technique non linéaire bien adaptée à la visualisation des ensembles de données de haute dimension.
- **ACP** : Une technique de réduction de dimensionnalité linéaire qui réduit la dimensionnalité de l'ensemble de données en transformant vers un nouvel ensemble de variables (composantes principales).

Après la réduction de dimensionnalité, des algorithmes de clustering ont été appliqués pour regrouper les points de données. Nous avons utilisé à la fois K-Means et le Clustering Agglomératif pour comparer les résultats.

### Collaboration
Ce projet a été un effort collaboratif. Chaque membre de l'équipe s'est concentré sur différents aspects :

- **Khallouq Youssef Amine** : A travaillé sur l'application de t-SNE pour la réduction de dimensionnalité et K-Means pour le clustering. Également responsable de la génération des graphiques de visualisation et de l'évaluation de la performance du clustering à l'aide de la Score de Silhouette, NMI et ARI.
- **MOUHASSINE Badreddine** : A appliqué l'ACP sur l'ensemble de données et mis en œuvre le Clustering Agglomératif, ainsi que les évaluations nécessaires.
- **Nabigh Mohamed** : S'est concentré sur UMAP pour la réduction de dimensionnalité et a aidé à affiner les paramètres du clustering pour améliorer la qualité des clusters.

Nous avons utilisé Git pour le contrôle de version et collaboré efficacement en utilisant des demandes d'extraction et des revues de code.

### Structure des Fichiers
- `main.py` : Le script principal où l'analyse de clustering est effectuée.
- `requirements.txt` : Liste tous les packages Python nécessaires pour le projet.

### Installation
Pour configurer l'environnement du projet :
```bash
git clone https://github.com/votre-repertoire.git
cd votre-repertoire
pip install -r requirements.txt
