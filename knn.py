from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # Importando o StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Carrega o dataset Iris
dataset = datasets.load_iris()

# Exibe os nomes das classes
print(dataset.target_names)

# Exibe os nomes das features
print(dataset.feature_names)

# Criando o objeto de normalização
scaler = StandardScaler()

# Separando os dados em treino e teste (60% treino, 40% teste)
X_train, X_temp, y_train, y_temp = train_test_split(
    dataset.data, dataset.target, test_size=0.4, random_state=42, stratify=dataset.target)

# Separando o conjunto temporário em validação e teste (50% para cada)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Aplicando a normalização
# Ajustamos o scaler apenas com os dados de treino
X_train_normalized = scaler.fit_transform(X_train)
# Para validação e teste, apenas aplicamos a transformação
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

print("Tamanho do conjunto de treino:", len(X_train))
print("Tamanho do conjunto de validação:", len(X_val))
print("Tamanho do conjunto de teste:", len(X_test))

# Função para contar as classes em cada conjunto
# def contar_classes(y):
#     classes, contagens = np.unique(y, return_counts=True)
#     return dict(zip(dataset.target_names, contagens))


# print("\nDistribuição das classes:")
# print("Dataset completo:", contar_classes(dataset.target))
# print("Conjunto de treino:", contar_classes(y_train))
# print("Conjunto de validação:", contar_classes(y_val))
# print("Conjunto de teste:", contar_classes(y_test))


k_values = []
accuracies_normalized_val = []  # acurácia na validação (normalizado)
accuracies_raw_val = []        # acurácia na validação (não normalizado)
accuracies_normalized_train = []  # acurácia no treino (normalizado)

# Loop pelos valores de k (apenas ímpares)
for k in range(1, 20, 2):
    # Cria dois modelos KNN com o valor atual de k
    knn_normalized = KNeighborsClassifier(n_neighbors=k)
    knn_raw = KNeighborsClassifier(n_neighbors=k)

    # Treina os modelos com os dados normalizados e não normalizados
    knn_normalized.fit(X_train_normalized, y_train)
    knn_raw.fit(X_train, y_train)

    # Faz previsões no conjunto de validação
    y_pred_normalized_val = knn_normalized.predict(X_val_normalized)
    y_pred_raw_val = knn_raw.predict(X_val)

    # Faz previsões no conjunto de treino (normalizado)
    y_pred_normalized_train = knn_normalized.predict(X_train_normalized)

    # Calcula as acurácias
    accuracy_normalized_val = accuracy_score(y_val, y_pred_normalized_val)
    accuracy_raw_val = accuracy_score(y_val, y_pred_raw_val)
    accuracy_normalized_train = accuracy_score(
        y_train, y_pred_normalized_train)

    # Armazena os resultados
    k_values.append(k)
    accuracies_normalized_val.append(accuracy_normalized_val)
    accuracies_raw_val.append(accuracy_raw_val)
    accuracies_normalized_train.append(accuracy_normalized_train)

    print(f"K = {k}")
    print(f"Acurácia Validação (normalizado): {accuracy_normalized_val:.4f}")
    print(f"Acurácia Validação (não normalizado): {accuracy_raw_val:.4f}")
    print(f"Acurácia Treino (normalizado): {accuracy_normalized_train:.4f}")
    print("-" * 50)

# Configurando o gráfico
plt.figure(figsize=(12, 6))

# Plotando as três curvas
plt.plot(k_values, accuracies_normalized_val, 'b-o',
         label='Validação (Normalizado)', linewidth=2)
plt.plot(k_values, accuracies_raw_val, 'r-o',
         label='Validação (Não Normalizado)', linewidth=2)
plt.plot(k_values, accuracies_normalized_train, 'g-o',
         label='Treino (Normalizado)', linewidth=2)

plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("Acurácia")
plt.title("Comparação da Acurácia do KNN em Diferentes Cenários")
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Melhorando a visualização
plt.ylim([0.8, 1.02])  # Ajusta o limite do eixo y para melhor visualização
plt.tight_layout()

plt.show()

# Escolhendo o melhor K baseado nos resultados anteriores
melhor_k = 9  # Você pode ajustar este valor baseado no gráfico que foi gerado

# Criando e treinando os modelos finais com o melhor K
knn_final_normalizado = KNeighborsClassifier(n_neighbors=melhor_k)
knn_final_raw = KNeighborsClassifier(n_neighbors=melhor_k)

# Treinando os modelos
knn_final_normalizado.fit(X_train_normalized, y_train)
knn_final_raw.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred_normalizado = knn_final_normalizado.predict(X_test_normalized)
y_pred_raw = knn_final_raw.predict(X_test)

# Calculando várias métricas de desempenho
print(f"\nResultados Finais com K = {melhor_k}")
print("\n=== Modelo com Dados Normalizados ===")
print("\nMatrix de Confusão:")
print(confusion_matrix(y_test, y_pred_normalizado))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_normalizado,
                            target_names=dataset.target_names, digits=4))

print("\n=== Modelo com Dados Não Normalizados ===")
print("\nMatrix de Confusão:")
print(confusion_matrix(y_test, y_pred_raw))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_raw,
                            target_names=dataset.target_names, digits=4))
