import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
from sklearn.metrics import roc_curve, auc 

df = pd.read_csv("../data/processed/gladiador_data_procesado.csv")
df['Survived'] = df['Survived'].astype(int)

X = df[["Wins", "Public Favor", "Allegiance Network_Strong"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

cv= KFold(10)
rfc = RandomForestClassifier(n_estimators=50, max_depth=5, max_features=3, random_state=42)
rfc.fit(X_train, y_train)
cv_rfc = cross_val_score(estimator=rfc, cv=cv, X= X, y= y, scoring="accuracy")
print(cv_rfc.mean())

#Realizo Predicciones
# Predicciones sobre el mismo conjunto de datos de entrenamiento
pred_rfc = rfc.predict(X_test)

#Obtengo las PROBABILIDADES de predicción
pred_proba = rfc.predict_proba(X_test)[:, 1]

print("--- Métricas de Clasificación ---")
print("Precisión (Accuracy): ", accuracy_score(y_test, pred_rfc))
print("Precisión (Clase 1): ", precision_score(y_test, pred_rfc))
print("Sensibilidad (Recall - Clase 1): ", recall_score(y_test, pred_rfc))
print("Puntuación F1 (Clase 1): ", f1_score(y_test, pred_rfc))
# Si pred_dtc son probabilidades, podrías calcular el ROC AUC:
print("ROC AUC: ", roc_auc_score(y_test, pred_proba))

# También puedes predecir para un nuevo dato, por ejemplo:
# Un personaje con 8 victorias, 0.7 de favor público y una red de lealtad fuerte (1)
new_data = pd.DataFrame([[8, 0.7, 1]], columns=["Wins", "Public Favor", "Allegiance Network_Strong"])
new_prediction = rfc.predict(new_data)

print(f"\nPredicción para un nuevo personaje (Wins: 8, Public Favor: 0.7, Allegiance Network_Strong: 1): {new_prediction[0]}")

if new_prediction[0] == 1:
    print("El modelo predice que este personaje Sobreviviría.")
else:
    print("El modelo predice que este personaje No Sobreviviría.")

with open("../models/rfc_model_final.pkl", "wb") as f:
    pickle.dump(rfc, f)

#MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_test, pred_rfc)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Sobrevive', 'Sobrevive'],
            yticklabels=['No Sobrevive', 'Sobrevive'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')

#CURVA ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
roc_auc = auc(fpr, tpr) # O puedes usar tu roc_auc_score previamente calculado

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) / Sensibilidad')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)

#FEATURE IMPORTANCE
feature_importances = rfc.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Importancia de las Características (Feature Importance)')
plt.xlabel('Importancia')
plt.ylabel('Característica')

#DISTRIBUCION DE PROBABILIDADES PREDICHAS
probabilities_class_0 = pred_proba[y_test == 0]
probabilities_class_1 = pred_proba[y_test == 1]

plt.figure(figsize=(10, 6))
sns.histplot(probabilities_class_0, color='red', kde=True, label='Clase Real: 0 (No Sobrevive)', stat='density', alpha=0.5)
sns.histplot(probabilities_class_1, color='green', kde=True, label='Clase Real: 1 (Sobrevive)', stat='density', alpha=0.5)
plt.title('Distribución de Probabilidades Predichas por Clase Real')
plt.xlabel('Probabilidad de Sobrevivir')
plt.ylabel('Densidad')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.close('all') 





