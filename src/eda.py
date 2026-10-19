import os
import pandas as pd
import numpy as np

# visualisation #
import matplotlib.pyplot as plt
import seaborn as sns
plt.show()

def perform_eda(df):
    
    print("\n=== INFORMATIONS GENERALES ===")
    print(df.info())
    print(df.head())

    print("\n=== STATISTIQUES DESCRIPTIVES ===")
    print(df.describe())
    
    print("\n=== VALEURS MANQUANTES ===")
    print(df.isnull().sum())
    
    print("\n=== TYPES DES VARIABLES ===")
    print(df.dtypes)
    
    # Distribution de la fréquance et de la cible montant de sinistres 
    sns.histplot(claimsauto['CLM_FREQ'], kde=True)
    sns.kdeplot(data=claimsauto, x="CLM_FREQ", fill=True, color="blue")
    plt.title("Distribution de CLM_FREQ (KDE)")
    plt.xlabel("CLM_FREQ")
    sns.histplot(claimsauto['CLM_AMT'], kde=True)
    
    sns.pairplot(claimsauto, height=6)
    plt.show()
    
    sns.heatmap(claimsauto.corr())
    plt.show()
    
    

claimsauto = pd.read_excel("C:/Users/Admin/Desktop/PROJET AUTO/data\Carclaimsforinsurance.xlsx")


claimsauto.describe()