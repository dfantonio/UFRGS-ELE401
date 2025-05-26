from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import joblib
import time
from datetime import timedelta

RANDOM_STATE = 2  # ou qualquer outro número

# Carrega o dataset de dígitos
dataset = datasets.load_digits()

# Exibe os nomes das classes
# print(dataset.target_names)

# Exibe os nomes das features
# print(dataset.feature_names)


# Primeira divisão: separa em (treino+validação) e teste
X_trainval, X_test, y_trainval, y_test = train_test_split(
    dataset.data, dataset.target,
    test_size=0.2,  # 20% para teste
    random_state=RANDOM_STATE,
    stratify=dataset.target
)

# Separando o conjunto temporário em treino e validação (50% para cada)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_trainval
)

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

accuracy_list = []
neuronios_list = []

# Iniciando o cronômetro total
tempo_total_inicio = time.time()

for first_layer in range(1, 201):
    # Configuração da rede neural
    clf = MLPClassifier(
        solver='lbfgs',
        hidden_layer_sizes=(first_layer,),
        activation='relu',
        max_iter=1000,
        random_state=RANDOM_STATE,
        alpha=0.0001,
        learning_rate='constant',
        learning_rate_init=0.001,
    )

    if (first_layer % 10 == 0):
        print(f"Treinando com {first_layer} neurônios")

    # Treinamento
    clf.fit(X_train, y_train)

    # Avaliação
    predictedY = clf.predict(X_val)
    accuracy = accuracy_score(y_val, predictedY)
    accuracy_list.append(accuracy)
    neuronios_list.append(first_layer)

# Calculando o tempo total
tempo_total = time.time() - tempo_total_inicio

# Criando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(range(1, 201), accuracy_list, 'b-', label='Acurácia')
plt.xlabel('Número de Neurônios na Camada Oculta')
plt.ylabel('Acurácia')
plt.title('Acurácia vs Número de Neurônios')
plt.grid(True)
plt.legend()

# Encontrando o melhor número de neurônios
best_neurons = np.argmax(accuracy_list) + 1
best_accuracy = max(accuracy_list)
plt.plot(best_neurons, best_accuracy, 'ro',
         label=f'Melhor: {best_accuracy:.4f} ({best_neurons} neurônios)')
plt.legend()

plt.savefig('acuracia_vs_neuronios.png')
plt.show()

# Salvando o gráfico
plt.close()

# Encontrando os 10 melhores resultados
resultados = list(zip(accuracy_list, neuronios_list))
resultados.sort(reverse=True)
top_10 = resultados[:10]

print(f"\nResultados do Treinamento:")
print(f"Melhor acurácia: {best_accuracy:.4f} com {best_neurons} neurônios")
print(f"Tempo total de treinamento: {timedelta(seconds=int(tempo_total))}")
print("\nTop 10 melhores resultados:")
for i, (acc, neurons) in enumerate(top_10, 1):
    print(f"{i}º lugar: Acurácia = {acc:.4f} com {neurons} neurônios")


# Treina com o melhor número de neurônios
clf = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(best_neurons,),
    activation='relu',
    max_iter=1000,
    random_state=RANDOM_STATE,
    alpha=0.0001,
    learning_rate='constant',
    learning_rate_init=0.001,
)
clf.fit(X_train, y_train)


# Salvando o modelo treinado
joblib.dump(clf, 'modelo_rede_neural.joblib')
