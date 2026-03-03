import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#1 Cargar datos a utilizar 
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)

print("Visualización inicial de los datos:\n")
print(df.head())

#df.info()

"""
1) Realizar el analisis exploratorio del dataset
2) Generar al menos 2 graficos que relacionen variables categoricas con la variable objetivo (Churn)
3) limpieza de los datos (por ejemplo eliminar el id del cliente ya que no aporta valor predictivo, convertir variables categoricas a numericas, etc)
4) convierte la variable objetivo churn a numerica (0 o 1)
5) aplica one hot encoding a las variables categoricas
6) divide el dataset en conjunto de entrenamiento (80%) y prueba (20%)


7) instacia de un un modelo de clasificacion 
8) entrenamiento del modelo con el conjunto de entrenamiento
Realiza la prediccion con el conjunto de prueba
9) evalua el modelo utilizando las metricas de accuracy, confusion matrix y classification report
"""