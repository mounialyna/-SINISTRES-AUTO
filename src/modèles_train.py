from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TweedieRegressor
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import joblib


def segmenter_decouper(x,y):
    mask_very_small = y <= 1000
    mask_small = (y> 1000) & (y < 4000)
    mask_large = y >= 4000

#Filtrer pour x
    x_very_small = x[mask_very_small]
    x_small = x[mask_small]
    x_large = x[mask_large]

# Filtrer pour y
    y_very_small = y[mask_very_small]
    y_small = y[mask_small]
    y_large = y[mask_large]

# train split 
# Segment 1 : Très petits sinistres (< 1 000)
    x_train_vs, x_test_vs, y_train_vs, y_test_vs = train_test_split(x_very_small, y_very_small, test_size=0.2, random_state=42)
    
    scaler_vs = StandardScaler()
    
    x_train_vs = scaler_vs.fit_transform(x_train_vs)
    x_test_vs = scaler_vs.transform(x_test_vs)
    

# Segment 2 : Petits sinistres [1 000 - 4 000[
    x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x_small, y_small, test_size=0.2, random_state=42)
   
    scaler_s = StandardScaler()
    x_train_s = scaler_s.fit_transform(x_train_s)
    x_test_s = scaler_s.transform(x_test_s)
    
    
# Segment 3 : Grands sinistres (>= 4 000)
    x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(x_large, y_large, test_size=0.2, random_state=42)
    
    scaler_l = StandardScaler()
    x_train_l = scaler_l.fit_transform(x_train_l)
    x_test_l = scaler_l.transform(x_test_l)

    return ((x_train_vs, x_test_vs, y_train_vs, y_test_vs),
          (x_train_s, x_test_s, y_train_s, y_test_s),
          (x_train_l, x_test_l, y_train_l, y_test_l),scaler_vs,scaler_s,scaler_l)

# --- 2. FONCTIONS D'ENTRAÎNEMENT PAR MODÈLE ---
    
   
def entrainer_tweedie(x_train, y_train):
    """Entraîne un modèle Tweedie pour les sinistres moyens."""
    model_s = TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000)
    model_s.fit(x_train, y_train)
    return model_s


def entrainer_lightgbm(x_train, y_train):
    """Entraîne un modèle LightGBM pour les grands sinistres."""
    model_l = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model_l.fit(x_train, y_train)
    return model_l

def entrainer_regression_logistique(x_train, y_train):
    """entrainer la regression logistque pour prédire à quel segment appartient un nouveau client"""
    y_train_classe = (y_train >= 4000).astype(int)
    log_reg = LogisticRegression(max_iter=1000,random_state=42)
    log_reg.fit(x_train, y_train_classe)
    return log_reg
   
    
    
    
# --- 3. FONCTION PRINCIPALE (Pipeline d'entraînement) ---


def entrainer_tous_les_modeles(x, y):
    """Fonction globale qui exécute la segmentation et l'entraînement complet."""
    print("Je suis entré dans entrainer_tous_les_modeles")
    seg_vs, seg_s, seg_l, scaler_vs, scaler_s, scaler_l = segmenter_decouper(x, y)

    x_train_vs, x_test_vs, y_train_vs, y_test_vs = seg_vs
    x_train_s, x_test_s, y_train_s, y_test_s = seg_s
    x_train_l, x_test_l, y_train_l, y_test_l = seg_l

    # Entraînement de chaque modèle sur son segment

    moyenne_vs = y_train_vs.mean()
    model_s = entrainer_tweedie(x_train_s, y_train_s)
    model_l = entrainer_lightgbm(x_train_l, y_train_l)
    scaler_log = StandardScaler()
    x_scaled = scaler_log.fit_transform(x)
    log_reg = entrainer_regression_logistique(x_scaled,y)
    
    print("Tous les modèles ont été entraînés avec succès !")
    
    import os
    os.makedirs("../models", exist_ok=True)
    print(os.getcwd())
   
    joblib.dump(moyenne_vs,"../models/moyenne_very_small.pkl")
    joblib.dump(model_s,"../models/model_small.pkl")
    joblib.dump(model_l,"../models/model_large.pkl")
    joblib.dump(log_reg,"../models/reg_model.pkl")
    joblib.dump(scaler_vs,"../models/scaler_vs.pkl")
    joblib.dump(scaler_s,"../models/scaler_s.pkl")
    joblib.dump(scaler_l,"../models/scaler_l.pkl")
    joblib.dump(scaler_log,"../models/scaler_log.pkl")
    # On retourne les modèles et les ensembles de test pour l'évaluation
    return {
        "very_small": (moyenne_vs, x_test_vs, y_test_vs),
        "small": (model_s, x_test_s, y_test_s),
        "large": (model_l, x_test_l, y_test_l),
        "log_reg": log_reg
        
    }
print("Modèle 'model_small.pkl' sauvegardé avec succès !")
# --- 4. sauvegarde des modèles ---
    

                                                            