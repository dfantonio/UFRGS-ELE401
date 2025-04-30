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


class my_quantized_DT():
    """
    Navigation through quantized Decision Tree
    Tiago Oliveira Weber 2023 (parte inicial)
    """

    def __init__(self, clf, bits):
        # clf is a sklearn tree classifier
        levels = 2**bits
        max_value = levels-1
        min_value = 0

        # quantizando parâmetros da árvore
        self.threshold = clf.tree_.threshold*max_value
        self.threshold = np.floor(self.threshold)
        self.threshold = self.threshold.astype(
            'int')         # value of comparison
        # sets to zero if threshold is negative
        self.threshold = np.maximum(self.threshold, 0)

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
            if x[self.feature[node]] < self.threshold[node]:
                next_node = self.children_left[node]
            else:
                next_node = self.children_right[node]

        return next_node, y


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

# Lista para armazenar as acurácias
acuracias = []
bits_range = range(1, 11)  # testa de 1 a 10 bits

# Para cada número de bits
for bits in bits_range:
    # Cria o modelo quantizado
    clf_qt = my_quantized_DT(modelo_final, bits)

    # Quantiza os dados
    max_value = 2**bits
    X_temp = X_test*max_value
    X_temp = np.floor(X_temp)
    X_qt = X_temp.astype('int')

    # Faz as previsões
    Y_qt = clf_qt.predict(X_qt)

    # Calcula a acurácia
    acuracia = accuracy_score(y_test, Y_qt)
    acuracias.append(acuracia/score_final)

    print(f"Bits: {bits}, Acurácia: {acuracia:.4f}")

# Plota os resultados
plt.figure(figsize=(10, 6))
plt.plot(bits_range, acuracias, 'b-o', linewidth=2)
plt.xlabel('Número de Bits')
plt.ylabel('Acurácia proporcional')
plt.title('Acurácia proporcional vs Número de Bits na Quantização')
plt.grid(True)
plt.xticks(bits_range)
plt.show()

# Encontra o melhor número de bits
melhor_bits = bits_range[np.argmax(acuracias)]
print(f"\nMelhor número de bits: {melhor_bits}")
print(f"Melhor acurácia: {max(acuracias):.4f}")
