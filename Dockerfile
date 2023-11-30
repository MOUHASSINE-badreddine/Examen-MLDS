# Utiliser une image de base Python
FROM python:3.8

# Définir le répertoire de travail dans le conteneur
WORKDIR /usr/src/app

# Copier les fichiers nécessaires dans le conteneur
COPY Requirements.txt ./
COPY main.py ./

# Installer les dépendances
RUN pip install -r Requirements.txt

# Commande à exécuter au démarrage du conteneur
CMD ["python", "./main.py"]
