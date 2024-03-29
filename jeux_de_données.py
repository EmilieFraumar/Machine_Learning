# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:43:13 2024

@author: anaca
"""

####### ancines donnes #########
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import arff
# # Parser un fichier de donnees au format arff
# # data est un tableau d’exemples avec pour chacun
# # la liste des valeurs des features
# #
# # Dans les jeux de donnees consideres :
# # il y a 2 features ( dimension 2)
# # Ex : [[ -0.499261 , -0. 0612356 ] ,
# # [ -1.51369 , 0. 265446 ] ,
# # [ -1.60321 , 0. 362039 ] , .....
# # ]
# #
# # Note : chaque exemple du jeu de donnees contient aussi un
# # numero de cluster . On retire cette information
# path = "./artificial/"
# 
# def data_set (nom):
#     databrut = arff.loadarff(open(path + nom +".arff", "r"))
#     datanp = [[x[0],x[1]] for x in databrut [0]]
#     datanp = np.array(datanp)
# 
# # Affichage en 2D
# # Extraire chaque valeur de features pour en faire une liste
# # Ex pour f0 = [ -0.499261 , -1.51369 , -1.60321 , ...]
# # Ex pour f1 = [ -0. 0612356 , 0.265446 , 0.362039 , ...]
# # print(datanp[0])
#     f0 = datanp [:,0] # tous les elements de la premiere colonne
#     f1 = datanp [:,1] # tous les elements de la deuxieme colonne
# # =============================================================================
# #     plt.scatter ( f0 , f1 , s=8 )
# #     plt.title (" Donnees initiales ")
# #     plt.show ()
# # =============================================================================
#     return f0, f1, datanp
######################################################




#################### neuveles donnes #####################

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff

# Parser un fichier de donnees au format arff
# data est un tableau d’exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2)
# Ex : [[ -0.499261 , -0. 0612356 ] ,
# [ -1.51369 , 0. 265446 ] ,
# [ -1.60321 , 0. 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
# path = "./artificial/"
path = "./dataset-rapport/"

def data_set(nom):
    # databrut = arff.loadarff(open(path + nom +".arff", "r"))
    databrut = np.loadtxt(path + nom + ".txt")  # Carregue os dados diretamente como um array numpy
    datanp = databrut[:, :2]  # Pegue as duas primeiras colunas como características
    # Affichage en 2D
    # Extraire chaque valeur de features pour en faire une liste
    # Ex pour f0 = [ -0.499261 , -1.51369 , -1.60321 , ...]
    # Ex pour f1 = [ -0. 0612356 , 0.265446 , 0.362039 , ...]
    # print(datanp[0])
    f0 = datanp[:, 0]  # tous les elements de la premiere colonne
    f1 = datanp[:, 1]  # tous les elements de la deuxieme colonne

    # =============================================================================
    #     plt.scatter ( f0 , f1 , s=8 )
    #     plt.title (" Donnees initiales ")
    #     plt.show ()
    # =============================================================================
    return f0, f1, datanp

