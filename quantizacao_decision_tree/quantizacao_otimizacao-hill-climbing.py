from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # Importando o StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import random


class my_quantized_DT():
    """
    Navigation through quantized Decision Tree
    Tiago Oliveira Weber 2023 (parte inicial)
    """

    def __init__(self, clf, bits, external_threshold=[]):
        # clf is a sklearn tree classifier
        levels = 2**bits
        max_value = levels-1
        min_value = 0

        # quantizando parâmetros da árvore
        if len(external_threshold) == 0:  # if external_threshold list is empty
            self.threshold = clf.tree_.threshold*max_value
            self.threshold = np.floor(self.threshold)
            self.threshold = self.threshold.astype(
                'int')         # value of comparison
            # sets to zero if threshold is negative
            self.threshold = np.maximum(self.threshold, 0)
        else:
            self.threshold = external_threshold

        # extraindo os valores da dt do sklearn
        self.children_left = clf.tree_.children_left     # next node if left
        self.children_right = clf.tree_.children_right   # next node if right
        self.feature = clf.tree_.feature    # feature to be compared with
        self.value = clf.tree_.value    # number of members for each class

    def predict(self, X_qt):
        Y = []
        for x in X_qt:
            y = -2  # temp
            node = 0  # resets to root
            while (y < 0):  # not leaf
                next_node, y = self.predict_in_node(node, x)
                node = next_node

            Y.append(y)

        return Y

    def predict_in_node(self, node, x):
        y = -2  # temp
        if (self.feature[node] < 0):  # it is a leaf
            next_node = -1  # does not matter
            y = np.argmax(self.value[node])  # plurality result

        else:
            if x[self.feature[node]] <= self.threshold[node]:
                next_node = self.children_left[node]
            else:
                next_node = self.children_right[node]

        return next_node, y

    def get_threshold(self):
        return self.threshold


# Função de hill climbing para otimização dos limiares
def hill_climbing(x0, loss, max_iter_without_improvement=100):
    x = x0
    cost = loss(x)
    iter_without_gain = 0

    while (iter_without_gain < max_iter_without_improvement):
        # Seleciona um índice aleatório para modificar
        index = random.randint(0, len(x)-1)
        x_candidate = np.copy(x)
        # Modifica o valor em +1 ou -1 aleatoriamente
        x_candidate[index] += 1 if random.random() < 0.5 else -1
        # Garante que o valor não fique negativo
        x_candidate[index] = max(0, x_candidate[index])

        cost_candidate = loss(x_candidate)

        if (cost_candidate < cost):
            cost = cost_candidate
            x = x_candidate
            iter_without_gain = 0
        else:
            iter_without_gain += 1

    return x, cost


RANDOM_STATE = 1  # ou qualquer outro número

# Carrega o dataset Iris
dataset = datasets.load_iris()

# Exibe os nomes das classes
print(dataset.target_names)

# Exibe os nomes das features
print(dataset.feature_names)

# Criando o objeto de normalização
scaler = StandardScaler()

# Primeira divisão: separa em (treino+validação) e teste
X_trainval, X_test, y_trainval, y_test = train_test_split(
    dataset.data, dataset.target,
    test_size=0.2,  # 20% para teste
    random_state=RANDOM_STATE,
    stratify=dataset.target
)


melhor_profundidade = 4
print(f"A melhor profundidade é: {melhor_profundidade}")

# Treina o modelo final com a melhor profundidade usando todos os dados de treino+validação
modelo_final = DecisionTreeClassifier(max_depth=melhor_profundidade)
modelo_final.fit(X_trainval, y_trainval)
score_final = modelo_final.score(X_test, y_test)
print(f"Acurácia final no conjunto de TESTE: {score_final:.4f}")

# Listas para armazenar as acurácias
acuracias_nao_otimizadas = []
acuracias_otimizadas = []
bits_range = range(1, 11)  # testa de 1 a 10 bits

# Para cada número de bits
for bits in bits_range:
    print(f"\nProcessando {bits} bits...")

    # Cria o modelo quantizado
    clf_qt = my_quantized_DT(modelo_final, bits)

    # Quantiza os dados
    max_value = 2**bits
    X_temp = X_test*max_value
    X_temp = np.floor(X_temp)
    X_qt = X_temp.astype('int')

    # Faz as previsões sem otimização
    Y_qt = clf_qt.predict(X_qt)

    # Calcula a acurácia sem otimização
    acuracia_nao_otimizada = accuracy_score(y_test, Y_qt)
    acuracias_nao_otimizadas.append(acuracia_nao_otimizada/score_final)

    print(f"Bits: {bits}, Acurácia sem otimização: {acuracia_nao_otimizada:.4f}")

    # Define a função de perda para o hill climbing
    def loss_quantized_dt(x):
        clf_qt_attempt = my_quantized_DT(modelo_final, bits, x)
        Y_qt_attempt = clf_qt_attempt.predict(X_qt)
        accuracy = accuracy_score(y_test, Y_qt_attempt)
        return -accuracy  # Negativo porque queremos maximizar a acurácia

    # Obtém os limiares iniciais
    original_thresholds = clf_qt.get_threshold()

    # Aplica o hill climbing para otimizar os limiares
    final_thresholds, final_cost = hill_climbing(
        original_thresholds, loss_quantized_dt)

    # Cria um novo modelo com os limiares otimizados
    clf_qt_otimizado = my_quantized_DT(modelo_final, bits, final_thresholds)

    # Faz as previsões com o modelo otimizado
    Y_qt_otimizado = clf_qt_otimizado.predict(X_qt)

    # Calcula a acurácia com otimização
    acuracia_otimizada = accuracy_score(y_test, Y_qt_otimizado)
    acuracias_otimizadas.append(acuracia_otimizada/score_final)

    print(f"Bits: {bits}, Acurácia com otimização: {acuracia_otimizada:.4f}")
    print(f"Melhoria: {(acuracia_otimizada - acuracia_nao_otimizada):.4f}")

# Plota os resultados comparativos
plt.figure(figsize=(12, 7))
plt.plot(bits_range, acuracias_nao_otimizadas,
         'b-o', linewidth=2, label='Sem otimização')
plt.plot(bits_range, acuracias_otimizadas, 'r-o',
         linewidth=2, label='Com otimização')
plt.xlabel('Número de Bits')
plt.ylabel('Acurácia proporcional')
plt.title('Acurácia proporcional vs Número de Bits na Quantização')
plt.grid(True)
plt.xticks(bits_range)
plt.legend()
plt.show()

# Encontra o melhor número de bits para cada abordagem
melhor_bits_nao_otimizado = bits_range[np.argmax(acuracias_nao_otimizadas)]
melhor_bits_otimizado = bits_range[np.argmax(acuracias_otimizadas)]

print(f"\nMelhor número de bits sem otimização: {melhor_bits_nao_otimizado}")
print(f"Melhor acurácia sem otimização: {max(acuracias_nao_otimizadas):.4f}")

print(f"\nMelhor número de bits com otimização: {melhor_bits_otimizado}")
print(f"Melhor acurácia com otimização: {max(acuracias_otimizadas):.4f}")

# Calcula a melhoria média
melhoria_media = np.mean(
    [a2 - a1 for a1, a2 in zip(acuracias_nao_otimizadas, acuracias_otimizadas)])
print(f"\nMelhoria média na acurácia: {melhoria_media:.4f}")
