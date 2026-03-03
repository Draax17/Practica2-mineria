import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



#1 Cargar datos a utilizar 
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)

print("Visualización inicial de los datos:\n")
print(df.head())
print(f"\nDimensiones del dataset: {df.shape}")


"""
1) Realizar el analisis exploratorio del dataset
2) Generar al menos 2 graficos que relacionen variables categoricas con la variable objetivo (Churn)
3) limpieza de los datos (por ejemplo eliminar el id del cliente ya que no aporta valor predictivo, convertir variables categoricas a numericas, etc)
4) convierte la variable objetivo churn a numerica (0 o 1)
5) aplica one hot encoding a las variables categoricas
6) divide el dataset en conjunto de entrenamiento (80%) y prueba (20%)


7) instacia de un un modelo de clasificacion 
8) entrenamiento del modelo con el conjunto de entrenamiento
9)Realiza la prediccion con el conjunto de prueba
10) evalua el modelo utilizando las metricas de accuracy, confusion matrix y classification report
"""

# 1) Análisis exploratorio básico
print("\nInformación general del dataset:")
print("############################################################")
df.info()
print("############################################################")

print("\nValores faltantes por columna:")
print(df.isnull().sum())

print("\nDistribución de la variable objetivo (Churn):")
print(df['Churn'].value_counts())


# 3) Limpieza de datos
df = df.drop(columns=['customerID'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

print("\nDataset después de limpieza (sin customerID y TotalCharges numérico):")
print("################################################################")
print(df.head())
print("################################################################")


# ----------------------------
# 2) Gráficas categóricas vs Churn
# ----------------------------
sns.set_style("whitegrid")

# Gráfica 1 mejorada: Género vs Churn (con porcentaje por barra)
plt.figure(figsize=(9, 6))
ax = sns.countplot(x='gender', hue='Churn', data=df, palette='Set2')
plt.title('Distribución de Churn por Género', fontsize=13)
plt.xlabel('Género')
plt.ylabel('Cantidad de Clientes')
plt.legend(title='Churn', loc='upper right')

total_gender = len(df)
for p in ax.patches:
	height = p.get_height()
	if height > 0:
		ax.annotate(
			f'{(height / total_gender) * 100:.1f}%',
			(p.get_x() + p.get_width() / 2, height),
			ha='center',
			va='bottom',
			fontsize=9,
			xytext=(0, 3),
			textcoords='offset points'
		)

plt.tight_layout()
plt.show()

# Gráfica 2: Tipo de contrato vs Churn
plt.figure(figsize=(10, 6))
ax2 = sns.countplot(
	x='Contract',
	hue='Churn',
	data=df,
	order=['Month-to-month', 'One year', 'Two year'],
	palette='Set1'
)
plt.title('Distribución de Churn por Tipo de Contrato', fontsize=13)
plt.xlabel('Tipo de Contrato')
plt.ylabel('Cantidad de Clientes')
plt.legend(title='Churn', loc='upper right')

for p in ax2.patches:
	height = p.get_height()
	if height > 0:
		ax2.annotate(
			f'{int(height)}',
			(p.get_x() + p.get_width() / 2, height),
			ha='center',
			va='bottom',
			fontsize=9,
			xytext=(0, 3),
			textcoords='offset points'
		)

plt.tight_layout()
plt.show()


# 4) Convertir variable objetivo Churn a numérica (0/1)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
print("\nChurn convertido a numérico (0=No, 1=Yes):")
print(df['Churn'].value_counts())


# 5) One-hot encoding para variables categóricas
X = df.drop(columns=['Churn'])
y = df['Churn']

categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"\nDimensiones de X después de One-Hot Encoding: {X_encoded.shape}")


# 6) División train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
	X_encoded,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y
)

print(f"Conjunto de entrenamiento: {X_train.shape}")
print(f"Conjunto de prueba: {X_test.shape}")


# Escalado de variables numéricas continuas
continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])


# 7) Instancia del modelo de clasificación
model = LogisticRegression(max_iter=1000, random_state=42)


# 8) Entrenamiento y predicción
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# 9) Evaluación del modelo
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n=== Evaluación del Modelo (Logistic Regression) ===")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)


# 10) Visualización de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - Logistic Regression')
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.tight_layout()
plt.show()



