import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(claimsauto):
    #copie du datafrae d'origine 
    df = claimsauto.copy()
    
    # Informations de controle
    print (claimsauto.columns)
    print (claimsauto.isnull().sum())
    categorical_cols= df.select_dtypes(include=['object','category']).columns
    print(categorical_cols)
    

    df=df.drop(["CLM_FREQ"], axis=1)
    
    # Encodage des variables catégorielles#
    # avant de passer à tweedie il faut encoder les variables catégorielles
    # pour les variables dont les valeurs sont yes or No 'PARENT1' et 'MSTATUS' les encoder manuellement#
    # pour la variable "BIRTH" qui est une date on extrait l'année, le mois et le jour en créant ces trois variables "year", "month" et "day"#
    df["MSTATUS"]=df["MSTATUS"].map({"Yes":1,"No":0})
    df["PARENT1"]=df["PARENT1"].map({"Yes":1,"No":0})
    df["GENDER"]=df["GENDER"].map({"M":1,"F":0})

#pour la colonne "BIRTH" on la sectionne en "year", "month" et "day"#
    df["BIRTH"].unique()
    df["year"]= df["BIRTH"].dt.year
    df["month"]= df["BIRTH"].dt.month
    df["day"]= df["BIRTH"].dt.day
    df=df.drop(["BIRTH"], axis=1)

#pour la variable "EDUCATION" on utilise "LabelEncoder" qui donne une valeur à chaque niveau d'éducation#
#pour la variable "CAR_TYPE" on utilise "LabelEncoder" qui donne une valeur à chaque niveau d'éducation#
    df["EDUCATION"].unique()
    df["CAR_TYPE"].unique()
    le=LabelEncoder()
    df["EDUCATION-encoded"]= le.fit_transform(df["EDUCATION"])
    df["CAR_TYPE-encoded"]= le.fit_transform(df["CAR_TYPE"])
    df["CAR_USE-encoded"]= le.fit_transform(df["CAR_USE"])
    print(df["MSTATUS"].unique())
    df.info()

    
    # Suppression des variables déja encodées
    df=df.drop(["EDUCATION","CAR_TYPE","CAR_USE"], axis=1)
    
    # One-Hot encoding de "OCCUPATION"
    df=pd.get_dummies(df, columns=["OCCUPATION"],drop_first=False)
    df=pd.get_dummies(df, columns=["URBANICITY"], drop_first=True)
    df.info()
    return df
   

    