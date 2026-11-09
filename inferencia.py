
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent

# 1. Carregar o modelo treinado
artefatos = joblib.load(
    BASE / "modelos" / "kmeans_boston.joblib"
)

modelo = artefatos["modelo"]
preprocessador = artefatos["preprocessador"]
colunas = artefatos["colunas"]
colunas_numericas = artefatos["colunas_numericas"]

# 2. Receber os dados da nova residência
print("=== DADOS DA NOVA RESIDÊNCIA ===")

dados = {}

for coluna in colunas:
    while True:
        try:
            valor = float(input(f"{coluna}: "))

            if coluna == "CHAS" and valor not in (0, 1):
                print("CHAS deve ser 0 ou 1.")
                continue

            dados[coluna] = valor
            break
        except ValueError:
            print("Digite um número válido.")

nova_residencia = pd.DataFrame([dados], columns=colunas)

# 3. Normalizar com o preprocessador treinado
dados_normalizados = preprocessador.transform(
    nova_residencia
)

# 4. Prever o cluster
grupo = int(modelo.predict(dados_normalizados)[0])

print("\n=== RESULTADO ===")
print(f"A residência pertence ao cluster: {grupo}")

# 5. Recuperar o centroide normalizado
centroide = modelo.cluster_centers_[grupo]

# 6. Desnormalizar as variáveis quantitativas
pipeline_numerico = (
    preprocessador.named_transformers_["numericas"]
)

scaler = pipeline_numerico.named_steps["scaler"]

quantidade_numericas = len(colunas_numericas)

centroide_numerico = scaler.inverse_transform(
    centroide[:quantidade_numericas].reshape(1, -1)
)[0]

# 7. Recuperar CHAS
# No centroide, CHAS representa a média dos valores 0 e 1
dados_centroide = dict(
    zip(colunas_numericas, centroide_numerico)
)

dados_centroide["CHAS"] = centroide[quantidade_numericas]

# 8. Exibir os valores na escala original
centroide_original = pd.Series(dados_centroide)
centroide_original = centroide_original[colunas]

print("\n=== CENTROIDE DESNORMALIZADO ===")
print(centroide_original.round(4).to_string())

print(
    "\nObservação: CHAS no centroide representa "
    "a proporção de registros com valor 1."
)