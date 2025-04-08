# prompt: Separe o dataset em 3 partes: 20% para validação; 20% para teste e 60% para treinamento

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

# Carrega o dataset Iris
dataset = datasets.load_iris()

# Exibe os nomes das classes
print(dataset.target_names)

# Exibe os nomes das features
print(dataset.feature_names)


# Separando os dados em treino e teste (80% treino, 20% teste)
X_train, X_temp, y_train, y_temp = train_test_split(
    dataset.data, dataset.target, test_size=0.2, random_state=42)

# Separando o conjunto temporário em validação e teste (50% para cada)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print("Tamanho do conjunto de treino:", len(X_train))
print("Tamanho do conjunto de validação:", len(X_val))
print("Tamanho do conjunto de teste:", len(X_test))
