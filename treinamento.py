
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Carregar a base
BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "HousingData.csv")

print("Dimensões da base:", df.shape)
print("Valores ausentes:")
print(df.isnull().sum())

# 2. Selecionar características
# MEDV não será usada para formar os clusters
X = df.drop(columns=["MEDV"])

colunas_numericas = [
    coluna for coluna in X.columns if coluna != "CHAS"
]

# 3. Preparar a normalização
# Variáveis quantitativas: mediana + StandardScaler
pipeline_numerico = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# CHAS: variável categórica binária (0 ou 1)
pipeline_categorico = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessador = ColumnTransformer([
    ("numericas", pipeline_numerico, colunas_numericas),
    ("categorica", pipeline_categorico, ["CHAS"])
])

X_normalizado = preprocessador.fit_transform(X)

# 4. Método do cotovelo
inercias = []
valores_k = range(1, 11)

for k in valores_k:
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    modelo.fit(X_normalizado)
    inercias.append(modelo.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(list(valores_k), inercias, marker="o")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inércia (distâncias quadráticas)")
plt.title("Método do Cotovelo - Boston Housing")
plt.xticks(list(valores_k))
plt.grid(True)

pasta_modelos = BASE / "modelos"
pasta_modelos.mkdir(exist_ok=True)

plt.savefig(pasta_modelos / "cotovelo.png")
plt.show()

# 5. Escolher o K com base no gráfico
k_otimo = int(input("Digite o K escolhido pelo cotovelo: "))

if k_otimo not in valores_k:
    raise ValueError("Escolha um K entre 1 e 10.")

# 6. Treinar o modelo definitivo
modelo_final = KMeans(
    n_clusters=k_otimo,
    random_state=42,
    n_init=10
)

modelo_final.fit(X_normalizado)

# 7. Salvar modelo e preprocessador
joblib.dump(
    {
        "modelo": modelo_final,
        "preprocessador": preprocessador,
        "colunas": X.columns.tolist(),
        "colunas_numericas": colunas_numericas
    },
    pasta_modelos / "kmeans_boston.joblib"
)

print("\nTreinamento concluído!")
print("Número de clusters:", k_otimo)
print("Modelo salvo em modelos/kmeans_boston.joblib")