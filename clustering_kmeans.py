# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:23:03 2024

@author: anaca
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster, metrics
from sklearn.metrics import davies_bouldin_score
from jeux_de_données import data_set

# Charger le jeu de données anciennes
# done_1 = "hypercube"
# done_2 = "2d-4c"
# done_3 = "square5"
# done_4 = "2d-3c-no123"

##### neuveles donnes 
done_1 = "x1"
done_2 = "x2"
done_3 = "x3"
done_4 = "x4"

def kMeans_silhouette(name):
    # Initialiser des variables pour le meilleur résultat
    f0, f1, datanp = data_set(name)
    best_k = 0
    best_silhouette_score = -1
    
    # Essayer différentes valeurs de k
    for k in range(2, 10):
        tps1 = time.time()
        
        # Appliquer l'algorithme KMeans
        model = cluster.KMeans(n_clusters=k, init='k-means++').fit(datanp)
        labels = model.labels_
    
        # Calculer le coefficient de silhouette
        silhouette_score = metrics.silhouette_score(datanp, labels, metric='euclidean')
    
        tps2 = time.time()
        iteration = model.n_iter_
    
# =============================================================================
#         # Afficher le résultat pour chaque configuration
#         plt.scatter(f0, f1, c=labels, s=8)
#         plt.title(f"Données après clustering Kmeans avec {k} clusters")
#         plt.show()
# =============================================================================
        runtime =round((tps2 - tps1) * 1000, 2)
        #print("Nombre de clusters =", k, ", Nombre d'itérations =", iteration, ", Runtime =", round((tps2 - tps1) * 1000, 2), "ms")
        #print("Coefficient de silhouette =", silhouette_score)
    
        # Mettre à jour la meilleure configuration si nécessaire
        if silhouette_score > best_silhouette_score:
            best_silhouette_score = silhouette_score
            best_k = k
            best_f0 = f0
            best_f1 = f1
            best_labels = labels
            runtime_best =runtime
    
    # Afficher la meilleure configuration
    print("Meilleur nombre de clusters =", best_k, ", Meilleur coefficient de silhouette =", best_silhouette_score,'runtime: ', runtime_best)
    
    # Afficher le résultat pour la çeilleure configuration
    plt.scatter(best_f0, best_f1, c=best_labels, s=8)
    plt.title(f"Données après clustering Kmeans_silhouette avec {best_k} clusters")
    plt.show()
 
    
 
def kMeans_davies_bouldin(name):
    # Initialiser des variables pour le meilleur résultat
    f0, f1, datanp = data_set(name)
    best_k = 0
    best_davies_bouldin_score = float('inf')
    
    # Essayer différentes valeurs de k
    for k in range(2, 10):
        tps1 = time.time()
        
        # Appliquer l'algorithme KMeans
        model = cluster.KMeans(n_clusters=k, init='k-means++').fit(datanp)
        labels = model.labels_
    
        # Calculer l'indice de Davies-Bouldin
        davies_bouldin_score_value = davies_bouldin_score(datanp, labels)
    
        tps2 = time.time()
        iteration = model.n_iter_
# =============================================================================
#     
#         # Afficher le résultat pour chaque configuration
#         plt.scatter(f0, f1, c=labels, s=8)
#         plt.title(f"Données après clustering Kmeans avec {k} clusters")
#         plt.show()
# =============================================================================
        
        print("Nombre de clusters =", k, ", Nombre d'itérations =", iteration, ", Runtime =", round((tps2 - tps1) * 1000, 2), "ms")
        print("Indice de Davies-Bouldin =", davies_bouldin_score_value)
    
        # Mettre à jour la meilleure configuration si nécessaire
        if davies_bouldin_score_value < best_davies_bouldin_score:
            best_davies_bouldin_score = davies_bouldin_score_value
            best_k = k
            best_f0 = f0
            best_f1 = f1
            best_labels = labels
    
    # Afficher la meilleure configuration
    print("Meilleur nombre de clusters =", best_k, ", Meilleur indice de Davies-Bouldin =", best_davies_bouldin_score)
    
    # Afficher le résultat pour la meilleure configuration
    plt.scatter(best_f0, best_f1, c=best_labels, s=8)
    plt.title(f"Données après clustering Kmeans_davies_bouldin avec {best_k} clusters")
    plt.show()    
 

kMeans_silhouette(done_1)

#### pour les anciennes donnes il faut cette partie ####
# kMeans_davies_bouldin(done_1)
# kMeans_silhouette(done_2)
# kMeans_davies_bouldin(done_2)
# kMeans_silhouette(done_3)
# kMeans_davies_bouldin(done_3)
# kMeans_silhouette(done_4)
# kMeans_davies_bouldin(done_4)