# Avaliação 2º Bimestre — K-Means Boston Housing

## Objetivo

Desenvolver um sistema de clusterização utilizando o algoritmo K-Means sobre a base Boston Housing, permitindo agrupar residências com características semelhantes.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Módulo de treinamento

O módulo `treinamento.py` realiza:

1. Leitura da base Boston Housing.
2. Tratamento de valores ausentes.
3. Normalização das variáveis quantitativas.
4. Tratamento da variável categórica CHAS.
5. Aplicação do método do cotovelo para escolha do número de clusters.
6. Treinamento do algoritmo K-Means.
7. Salvamento do modelo e do preprocessador.

## Módulo de inferência

O módulo `inferencia.py` permite:

1. Informar os dados de uma nova residência.
2. Aplicar o preprocessamento utilizado no treinamento.
3. Identificar o cluster correspondente.
4. Exibir o centroide com os valores na escala original.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o treinamento:

```bash
python treinamento.py
```

Depois, execute a inferência:

```bash
python inferencia.py
```

## Base de dados

Boston Housing, contendo 506 registros e 14 colunas.

A coluna MEDV foi excluída das características de clusterização.

## Observações

O algoritmo K-Means é um método de aprendizado não supervisionado. Os clusters são definidos pela similaridade entre as características das residências.

O método do cotovelo é utilizado para auxiliar na escolha do número de clusters.