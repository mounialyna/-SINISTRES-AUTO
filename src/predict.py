import joblib

# charger les modèles
def charger_modeles():
    moyenne_vs = joblib.load("../models/moyenne_very_small.pkl")
    model_s = joblib.load("../models/model_small.pkl")
    model_l = joblib.load("../models/model_large.pkl")
    log_reg = joblib.load("../models/reg_model.pkl")
    scaler_log = joblib.load("../models/scaler_log.pkl")
    scaler_s = joblib.load("../models/scaler_S.pkl")
    scaler_l = joblib.load("../models/scaler_l.pkl")
    
    return moyenne_vs, model_s, model_l, log_reg, scaler_log,scaler_s,scaler_l

def predict(new_client):
    moyenne_vs,model_s, model_l,log_reg, scaler_log,scaler_s,scaler_l = charger_modeles()
    # probabilité d'être un grand sinistre
    new_client_scaled = scaler_log.transform(new_client)
    proba = log_reg.predict_proba(new_client_scaled)[:, 1][0]
    threshold=0.6
    if proba >= threshold:
        new_client_l=scaler_l.transform(new_client)
        prediction = model_l.predict(new_client_l)[0]
        return {"segment": "large","probabilite": proba,"montant_predit": prediction,"modele_utilise": "LightGBM"}
    else: 
        new_client_s=scaler_s.transform(new_client)
        prediction = model_s.predict(new_client_s)[0]
        
        print("Prediction brute :", prediction)
        return {"segment": "small","probabilite": proba,"montant_predit": prediction,"modele_utilise": "Tweedie"}
    
# faire un test de prediction avec une ligne de ma base de données 

import pandas as pd
from preprocessing import preprocess_data
from feature_selection import feature_selection 
df=pd.read_excel("C:/Users/Admin/Desktop/PROJET 2026/Carclaimsforinsurance.xlsx")

df = preprocess_data(df)
x,y = feature_selection(df)
vraivaleur = y.iloc[1] 
print("vraivaleur :", vraivaleur) 
new_client=x.iloc[[7644]]
resultat = predict(new_client)
print(resultat)

    