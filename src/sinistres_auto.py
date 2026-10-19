#!/usr/bin/env python
# coding: utf-8

# # MODELISATION PREDICTIVE DES SINISTRES AUTO

# # 1/ Présentation du projet

# # analyse des sinistres auto et modélisation avec un modèle GLM "TWEEDIE" puis un modèle ML "LightGBM" et comparaison des deux modèles# 

# In[1]:
    # imports standrds
import os
import pandas as pd


from eda import perform_eda
from preprocessing import preprocess_data
from feature_selection import feature_selection
from modèles_train import entrainer_tous_les_modeles

def main ():
    print("== Début du pipeline sinistres_auto ==")
# 1.chargement des données
    print("Chargement des données")
    df=pd.read_excel("C:/Users/Admin/Desktop/PROJET 2026/Carclaimsforinsurance.xlsx")

# 2.Analyse exploratrice des données
    print("eda")
    perform_eda(df)
    
# 3.Preprocessing
    print("preprocessing data")
    df = preprocess_data(df)
    
# 4. Selection des variables
    print("feature selection") 
    x,y = feature_selection(df)
    
# 5. Entrainement des modèles
    print("Entrainer tous les modèles")  
    resultats = entrainer_tous_les_modeles(x,y)
    return resultats

if __name__ == "__main__":
    resultas = main()
    print("pipeline terminé avec succès.")
    





