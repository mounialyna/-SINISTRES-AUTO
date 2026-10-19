
import pandas as pd

def feature_selection(df):
  x = df.drop("CLM_AMT", axis=1)
  y = df["CLM_AMT"]
  
  variables_a_supprimer = ["CAR_USE-encoded","OCCUPATION_Student","GENDER","OCCUPATION_Clerical","OCCUPATION_Lawyer","OCCUPATION_Manager","OCCUPATION_Doctor"]
  x = x.drop (variables_a_supprimer, axis=1)
  
  return (x,y) 
