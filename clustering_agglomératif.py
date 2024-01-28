import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from jeux_de_données import data_set
from sklearn import cluster, metrics
import time


def evaluate_clustering(labels_pred):
    silhouette_score = metrics.silhouette_score(datanp, labels_pred)
    return silhouette_score



############anciennes donnes ######################

# # Donnees dans datanp
# donnes = ['hypercube', 'shapes','atom','square5']
# threshold = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [5, 8, 10, 13, 17],[1, 2, 3, 4, 5]]
# linkage_methods = ['single', 'average', 'complete', 'ward']
# i=0

# # set distance_threshold (0 ensures we compute the full tree )
# print ("Changement du distance_threshold")

# for donne in donnes:
#     f0, f1, datanp = data_set(donne)
#     print ("")
#     print (donne)
#     print ("")

#     distance_threshold =  threshold[i]
#     i=i+1
#     best_linkage = 'single'
    
#     best_silhouette_score = float('-inf')
#     best_silhouette_params = None
#     best_silhouette_labels = None

#     best_davies_bouldin_score = float('inf')
#     best_davies_bouldin_params = None
#     best_davies_bouldin_labels = None
    
#     for linkage in linkage_methods:
#         best_silhouette_score_avant = 0
        
#         for distance in distance_threshold:
#             tps1 = time.time()
#             model = cluster.AgglomerativeClustering(distance_threshold = distance ,linkage = linkage, n_clusters = None )
#             model = model.fit(datanp)
#             tps2 = time.time()
#             labels = model.labels_
#             k = model.n_clusters_
#             if k == 1:
#                 continue
#             leaves = model.n_leaves_
#             runtime = round (( tps2 - tps1 )*1000 , 2 )
#             silhouette_score = evaluate_clustering(labels)
#             if silhouette_score > best_silhouette_score:
#                 best_silhouette_score = silhouette_score
#                 best_silhouette_params = {'threshold': distance, 'n_clusters': k, 'Silhouette': best_silhouette_score, 'runtime = ': runtime,'linkage = ': linkage}
#                 best_silhouette_labels = labels
#                 best_linkage = linkage
            
#     # Affichage clustering Silhouette
#     plt.scatter( f0 , f1 , c=best_silhouette_labels , s=8 )
#     plt.title(" Resultat du clustering " + donne )
#     plt.show()
#     print(f"Best params: {best_silhouette_params}")
    
#     linked_mat = shc.linkage(datanp , best_linkage)
#     plt.figure(figsize =(12 , 12))
#     shc.dendrogram( linked_mat,
#     orientation ='top',
#     distance_sort ='descending',
#     show_leaf_counts = False )
#     plt.title(donne)
#     plt.show()






###################################### pour les neuveles donnes ###########################

donnes = ['y1']
linkage_methods = ['single', 'average', 'complete', 'ward']

print ("Changement du k")
# set the number of clusters
k_range=[1, 2, 3, 4, 5, 6, 7, 8, 9]

for donne in donnes:
    f0, f1, datanp = data_set(donne)
    print ("")
    print (donne)
    print ("")
    best_linkage = 'single'
    
    best_silhouette_score = float('-inf')
    best_silhouette_params = None
    best_silhouette_labels = None

    best_davies_bouldin_score = float('inf')
    best_davies_bouldin_params = None
    best_davies_bouldin_labels = None
    
    for linkage in linkage_methods:
        best_silhouette_score_avant = 0
        for k in k_range:
            tps1 = time.time()
            model = cluster.AgglomerativeClustering(linkage = linkage, n_clusters = k)
            model = model.fit(datanp)
            tps2 = time.time()
            labels = model.labels_
            k = model.n_clusters_
            if k == 1:
                continue
            leaves = model.n_leaves_
            runtime = round (( tps2 - tps1 )*1000 , 2 )
            silhouette_score = evaluate_clustering(labels)
            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_silhouette_params = {'n_clusters': k, 'Silhouette': best_silhouette_score, 'runtime = ': runtime,'linkage = ': linkage}
                best_silhouette_labels = labels
                best_linkage = linkage

            
    # Affichage clustering Silhouette
    plt.scatter( f0 , f1 , c=best_silhouette_labels , s=8 )
    plt.title(" Resultat du clustering " + donne )
    plt.show()
    print(f"Best params: {best_silhouette_params}")



