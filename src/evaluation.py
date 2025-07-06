import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

df = pd.read_csv("../data/processed/gladiador_data_procesado.csv")
df['Survived'] = df['Survived'].astype(int)

X = df[["Wins", "Public Favor", "Allegiance Network_Strong"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelo Bagging Classifier
estimator = DecisionTreeClassifier(max_depth=3, random_state=42)

bag_clf = BaggingClassifier(
    estimator = estimator,
    n_estimators=300, # Cantidad de modelos
    max_samples=100, # Muestras utilizadas en boostrapping
    bootstrap=True, # Usamos boostrapping
    max_features = 3, # Features que utiliza en el boostrapping. Cuanto más bajo, mejor generalizará y menos overfitting
    random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
cv= KFold(10)
cv_bc = cross_val_score(estimator=bag_clf, X=X, y=y, cv=cv, scoring="accuracy")
print(cv_bc.max())

#Modelo Random Forest
cv= KFold(10)
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=3, random_state=42)
rfc.fit(X_train, y_train)
cv_rfc = cross_val_score(estimator=rfc, cv=cv, X= X, y= y, scoring="accuracy")
print(cv_rfc.mean())

#Modelo AdaBost Classifier
estimator = DecisionTreeClassifier(max_depth=1)

ada_clf = AdaBoostClassifier(estimator = estimator,
                             n_estimators=200,
                             learning_rate=0.5, #va de 0 a 1 /es cuanto e tiene en cuenta las predicciones del valor anterior
                             random_state=42)

cv_ada = cross_val_score(estimator=ada_clf, X=X, y=y, cv=10, scoring="accuracy")
print(cv_ada.mean())

#Modelo Gradient Boosting Classifier
gbc = GradientBoostingClassifier(max_depth=5,n_estimators=100, random_state=42, learning_rate=0.5)
cv_gbc = cross_val_score(estimator=gbc, X=X, y=y,cv=10,scoring="accuracy")
print(cv_gbc.mean())

#Modelo XGB Classifier
xgb = XGBClassifier(n_estimators = 100, random_state = 10, learning_rate = 0.5, eval_metric='logloss')
cv_xgb = cross_val_score(estimator=xgb, X=X, y=y, cv= 10, scoring="accuracy")
print(cv_xgb.mean())

#Modelo Regresión Logística
lr_model = LogisticRegression(random_state=42, solver='liblinear')
cv_lr = cross_val_score(estimator=lr_model, X=X, y=y, cv= 10, scoring="accuracy")
print(cv_lr.mean())

#Modelo Decission Tree Classifier
dtc_model = DecisionTreeClassifier(random_state=42)
cv_dtc = cross_val_score(estimator=dtc_model, cv=10, X= X, y= y, scoring="accuracy")
print(cv_dtc.mean())

#Primeros Resultados
modelos = [cv_bc.mean(), cv_rfc.mean(), cv_ada.mean(), cv_gbc.mean(), cv_xgb.mean(), cv_lr.mean(), cv_dtc.mean()]
df_resultados = pd.DataFrame(modelos, columns=["Accuracy"], index=["BaggingClassifier", "RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier", "XGBClassifier", "LogisticRegression", "DecissionTreeClassifier"])

df_resultados.sort_values("Accuracy", ascending=False)

#Hiperparametrización
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

rfc = RandomForestClassifier()
parametros={
    "n_estimators":[50,100,150],
    "max_depth":[3,5,7,10,12],
    "max_features":[2,3,4],
    "bootstrap":[True, False]
}

gs_rfc = GridSearchCV(rfc, parametros, scoring="accuracy", cv = 5, verbose=3, n_jobs=1)
gs_rfc.fit(X_train, y_train)
print(gs_rfc.best_estimator_)
print(gs_rfc.best_score_)

#Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", BaggingClassifier())

])

parametros = [
    {
        "scaler":[None, StandardScaler(), MinMaxScaler()],
        "model": [BaggingClassifier(random_state=42)],
        "model__n_estimators":[50,100,150],
        "model__max_samples": [0.5, 1]
    },
    {
        "scaler":[None, StandardScaler(), MinMaxScaler()],
        "model": [RandomForestClassifier(random_state=42)],
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [3, 5, 7, 10, 12]
    },
    {
        "scaler":[None, StandardScaler(), MinMaxScaler()],
        "model": [AdaBoostClassifier(random_state=42)],
        "model__n_estimators":[50,100,150],
        "model__learning_rate":[0.5, 0.75]
    },
    {
        "scaler":[None, StandardScaler(), MinMaxScaler()],
        "model": [GradientBoostingClassifier(random_state=42)],
        "model__n_estimators":[50,100,150],
        "model__learning_rate":[0.5, 0.75]
    },
    {
        "scaler":[None, StandardScaler(), MinMaxScaler()],
        "model":[XGBClassifier(random_state=42)],
        "model__n_estimators":[50, 100, 150],
        "model__learning_rate":[0.5, 0.75],
        "model__max_depth":[3,5,7,12]
    },
    {
        "scaler":[None, StandardScaler(), MinMaxScaler()],
        "model":[LogisticRegression(random_state=42)],
    },
    {
        "scaler":[None, StandardScaler(), MinMaxScaler()],
        "model":[DecisionTreeClassifier(random_state=42)],
        "model__max_depth": [3, 5, 7, 10, 12]
    }
]

gs_final = GridSearchCV(pipeline, parametros, cv = 10, scoring="accuracy", n_jobs=1, verbose=3)
gs_final.fit(X_train, y_train)

# Imprimo el mejor score y los mejores parámetros del GridSearchCV final
print(f"Mejor score del GridSearchCV final: {gs_final.best_score_}")
print(f"Mejor estimador del GridSearchCV final: {gs_final.best_estimator_}")

# Guardo el mejor modelo (pipeline) en la carpeta 'models'
best_model_filename = '../models/best_gladiator_survival_model.pkl' # Define la ruta y el nombre del archivo
with open(best_model_filename, 'wb') as file:
    pickle.dump(gs_final.best_estimator_, file)

print(f"Mejor modelo guardado en {best_model_filename}")

