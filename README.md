\# Prédiction des sinistres auto avec un pipeline hybride Tweedie + LightGBM

\## Objectif du projet

**Développer un pipeline intelligent de prédiction des sinistres automobiles, combinant modèles statistiques et techniques de machine learning.**

Ce projet a pour objectif de prédire le montant d'un sinistre automobile à partir des caractéristiques d'un assuré et de son véhicule.



L'approche développée combine les méthodes traditionnelles de l'actuariat avec des techniques modernes de Machine Learning afin d'obtenir des prédictions plus précises et adaptées aux différents profils de clients.



Après avoir constaté les limites d'un modèle unique appliqué à l'ensemble du portefeuille, une approche hybride a été conçue :



\- Segmentation des sinistres en plusieurs groupes ;

\- Régression logistique pour identifier le segment d'un nouveau client ;

\- Tweedie Regressor pour certains segments ;

\- LightGBM pour les segments les plus complexes ;

\- Utilisation d'un coût moyen pour les très petits sinistres.



Cette démarche reproduit une problématique réelle rencontrée en assurance : différents profils de sinistres nécessitent différentes méthodes de modélisation.

\---

\## Contexte métier

Dans le domaine de l’assurance la prédiction des sinistres est essentielle pour 

\- \*\* Tarification\*\* : ajuster les primes en fonction du risque réel. 

\- \*\* Gestion des risques\*\* : ajuster les primes en fonction du risque réel. 

\- \*\* Optimisation des primes\*\* : proposer des tarifs compétitifs tout en restant rentables.

\- \*\* Segmentation clients\*\* : adapter les méthodes de calcul selon les profils pour plus de précision.

Ce projet illustre une démarche complète : partir d’essais classiques, constater leurs limites, puis concevoir une solution hybride et automatisée qui reflète les réalités du métier. 

L'objectif est de construire un système capable d'estimer automatiquement le montant probable d'un futur sinistre tout en tenant compte de l'hétérogénéité des profils clients.


\##  Étapes du projet


\##  Données utilisées

\- Source : [Kaggle]

\- Dataset : Auto Insurance Claims

\- Variable cible : 

CLM\_AMT : montant du sinistre 

-Exemples de variables explicatives :

\- AGE

\- INCOME

\- HOME\_VAL

\- BLUEBOOK

\- CAR\_AGE

\- YOJ

\- TIF

\- MVR\_PTS

\- TRAVTIME

\- CAR\_USE

\- EDUCATION

\- OCCUPATION



\---

\## Architecture du projet

PROJET\_AUTO/

` `├── data/

│

├── notebook/

│ └── sinistres\_auto.ipynb

│

├── src/

│ ├── eda.py

│ ├── preprocessing.py

│ ├── feature\_selection.py

│ ├── modèles\_train.py

│ ├── predict.py

│ └── sinistres\_auto.py

│

├── models/

│ └── \*.pkl

│

├── app.py

├── requirements.txt

├── Dockerfile

├── .gitignore

\*── README.md

\```

\## Pipeline\*Machine Learning

\### 1. Analyse exploratoire (EDA)

Une analyse exploratoire a été réalisée afin de comprendre la structure du portefeuille automobile :

- Étude de la distribution des montants de sinistres (CLM\_AMT) ;
- Analyse de la fréquence des sinistres (CLM\_FREQ) ;
- Analyse des corrélations entre variables ;
- Détection des variables les plus influentes ;
- Visualisation des distributions et des relations entre variables.

Cette étape a mis en évidence une forte asymétrie des montants de sinistres ainsi qu'une importante hétérogénéité du portefeuille.

\- Analyse de la \*istribution des sinistres

\- Étude \*es variables explicatives

\- Recher\*he de valeurs aberrantes

\- Analyse\*des corrélations

\### 2. Prétraitement des données

Les données ont ensuite été préparées pour la modélisation :

- Transformation des variables dates ;
- Encodage des variables catégorielles ;
- One-Hot Encoding des variables OCCUPATION et URBANICITY ;
- Encodage des variables binaires ;
- Suppression des variables devenues inutiles après transformation ;
- Standardisation des variables numériques continues.

\### Premiers entrainement des modèles

Un premier modèle Tweedie Regressor a été entraîné sur l'ensemble du portefeuille afin de prédire directement les montants de sinistres.

Bien que les résultats aient montré une bonne capacité de généralisation, l'analyse des résidus a révélé une difficulté à prédire correctement les sinistres les plus élevés. Le modèle avait tendance à sous-estimer les valeurs extrêmes.

\### Segmentation des données

Pour mieux capturer l'hétérogénéité des montants, les sinistres ont été segmentés en plusieurs catégories :

- Very Small Claims (< 1000)
- Small Claims (1000 à 4000)
- Large Claims (≥ 4000)

Cette segmentation permet d'appliquer une stratégie adaptée à chaque population de sinistres.

\- Entraînement de Tweedie sur les deux segments →résultat Tweedie adapté sur le premier segment

\- Entraînement de LightGBM sur les deux segments → résultat LightGBM adapté sur le deuxième segment

\### Construction d’un Pipeline hybride

Une approche hybride a ensuite été développée :

- Les Very Small Claims sont estimés à partir d'un coût moyen ;
- Les Small Claims sont modélisés par un Tweedie Regressor ;
- Les Large Claims sont modélisés par un LightGBM Regressor.

Cette combinaison permet de tirer parti à la fois des modèles actuariels traditionnels et des algorithmes de Machine Learning. 

\#### Résultats 

Amélioration significative des performances grâce à la segmentation

\- Pipeline automatisé permettant d’adapter le modèle au profil du client

\- Approche robuste combinant actuariat (Tweedie) et machine learning (LightGBM)

\- Gestion spécifique des petits sinistres hétérogènes par une approche simple et efficace

\### Classification automatique de nouveaux clients

Une régression logistique est utilisée pour estimer la probabilité qu'un nouveau client appartienne à la catégorie des grands sinistres.

\- Application du modèle correspondant (Tweedie ou LightGBM) selon le segment identifié 

\### Selection des variables

Après construction du pipeline segmenté, une étape de sélection des variables a été réalisée à l'aide d'un Random Forest.

Suite au premier entraînement, une anomalie a été mise en évidence lors des prédictions individuelles : le modèle Tweedie produisait des résultats quasi identiques pour des clients aux profils pourtant bien distincts (que ce soit pour un grand ou un petit sinistre), révélant un manque de discrimination et la présence de bruit dans les données.

Pour corriger ce défaut :**Analyse d'importance :** Utilisation d'un modèle Random Forest pour identifier et isoler les variables réellement contributives.

\### **Réentraînement des modèles**

Les modèles Tweedie, LightGBM et Régression Logistique ont ensuite été réentraînés sur le jeu de variables sélectionnées afin d'obtenir la version finale du pipeline.

\### **Prédiction automatique**

Pour chaque nouveau client :

1. Prétraitement des données ;
1. Sélection des variables ;
1. Estimation de la probabilité d'un grand sinistre par la régression logistique ;
1. Sélection automatique du modèle approprié ;
1. Prédiction du montant du sinistre.

Le système fonctionne ainsi comme un moteur de décision entièrement automatisé permettant d'assigner à chaque assuré la méthode de prédiction la plus adaptée à son profil.

**## Technologies utilisées**

\- Python

\- Pandas

\- NumPy

\- \*cikit-Learn

\- LightGBM

\- Joblib

\- Git

\- GitHub

\---

\## **Résultats**

Le projet a permis :

\- Amélioration significative des performances grâce à la segmentation

\- Pipeline automatisé permettant d’adapter le modèle au profil du client

\- Approche robuste combinant actuariat (Tweedie) et machine learning (LightGBM)

\- Gestion spécifique des petits sinistres hétérogènes par une approche simple et efficace

\- Construction d’un pipeline complet de prédiction.

\---

\## Utilisation 

L'objectif est de construire un système capable d'estimer automatiquement le montant probable d'un futur sinistre tout en tenant compte de l'hétérogénéité des profils clients.

\## Versionnement et Collaboration

\- Le projet est entièrement \*\*versionné avec Git\*\* pour assurer la traçabilité des modifications.

\- Chaque étape clé (prototype, ajout du Dockerfile, pipeline CI/CD) est commitée avec un message claire.  

\- Le dépôt GitHub sert de \*\*source unique de vérité\*\*, facilitant la collaboration et la visibilité.

\## Installation 

\### Prérequis

\- Python 3.11.15

\- Git installé sur la machine

\- Docker 




