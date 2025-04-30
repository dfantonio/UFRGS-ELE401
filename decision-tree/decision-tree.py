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
import graphviz
from sklearn.tree import plot_tree


RANDOM_STATE = 1  # ou qualquer outro número

# Carrega o dataset Iris
dataset = datasets.load_iris()

# Exibe os nomes das classes
print(dataset.target_names)

# Exibe os nomes das features
print(dataset.feature_names)

# Criando o objeto de normalização
scaler = StandardScaler()

# # Separando os dados em treino e teste (60% treino, 40% teste)
# X_train, X_temp, y_train, y_temp = train_test_split(
#     dataset.data, dataset.target, test_size=0.4, random_state=22, stratify=dataset.target)

# # Separando o conjunto temporário em validação e teste (50% para cada)
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, random_state=22, stratify=y_temp)

# Primeira divisão: separa em (treino+validação) e teste
X_trainval, X_test, y_trainval, y_test = train_test_split(
    dataset.data, dataset.target,
    test_size=0.2,  # 20% para teste
    random_state=RANDOM_STATE,
    stratify=dataset.target
)

# Definir possíveis profundidades para testar
max_depths = range(1, 11)  # testa profundidades de 1 a 10

# Para cada profundidade
scores_por_profundidade_val = []  # acurácias na validação
scores_por_profundidade_train = []  # acurácias no treino
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for depth in max_depths:
    scores_fold_val = []
    scores_fold_train = []
    for train_idx, val_idx in kf.split(X_trainval):
        # Separa os dados
        X_train_fold = X_trainval[train_idx]
        X_val_fold = X_trainval[val_idx]
        y_train_fold = y_trainval[train_idx]
        y_val_fold = y_trainval[val_idx]

        # Treina o modelo
        tree = DecisionTreeClassifier(max_depth=depth,
                                      random_state=RANDOM_STATE)
        tree.fit(X_train_fold, y_train_fold)

        # Avalia na validação
        score_val = tree.score(X_val_fold, y_val_fold)
        scores_fold_val.append(score_val)

    # Avalia no treino
    score_train = tree.score(X_trainval, y_trainval)
    scores_fold_train.append(score_train)

    # Guarda as médias das scores para esta profundidade
    scores_por_profundidade_val.append(np.mean(scores_fold_val))
    scores_por_profundidade_train.append(np.mean(scores_fold_train))

# Encontra a melhor profundidade (baseado na validação)
melhor_profundidade = max_depths[np.argmax(scores_por_profundidade_val)]
print(f"A melhor profundidade é: {melhor_profundidade}")

# Plota os resultados
plt.figure(figsize=(10, 6))
plt.plot(max_depths, scores_por_profundidade_val, 'b-o',
         label='Validação', linewidth=2)
plt.plot(max_depths, scores_por_profundidade_train, 'r-o',
         label='Treino', linewidth=2)

plt.xlabel('Profundidade máxima')
plt.ylabel('Acurácia')
plt.title('Acurácia vs Profundidade da Árvore: Treino e Validação')
plt.grid(True)
plt.xticks(range(1, 11, 2))
plt.legend()
plt.show()

# Treina o modelo final com a melhor profundidade usando todos os dados de treino+validação
modelo_final = DecisionTreeClassifier(max_depth=melhor_profundidade)
modelo_final.fit(X_trainval, y_trainval)

# Avalia no conjunto de teste
score_final = modelo_final.score(X_test, y_test)
print(f"Acurácia final no conjunto de teste: {score_final:.4f}")

# Relatório detalhado
y_pred = modelo_final.predict(X_test)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, digits=3))

# # Cria a visualização da árvore
# dot_data = export_graphviz(
#     modelo_final,
#     feature_names=dataset.feature_names,  # Nomes das features
#     class_names=dataset.target_names,     # Nomes das classes
#     filled=True,                          # Preenche os nós com cores
#     rounded=True,                         # Cantos arredondados
#     special_characters=True,              # Caracteres especiais
#     impurity=False                        # Não mostra a impureza
# )

# # Cria o gráfico
# graph = graphviz.Source(dot_data)

# # Salva o gráfico em um arquivo
# graph.render("arvore_decisao", format="png", cleanup=True)

# # Mostra o gráfico
# graph

# Matplotlib
# plt.figure(figsize=(20, 10))
# plot_tree(
#     modelo_final,
#     feature_names=dataset.feature_names,
#     class_names=dataset.target_names,
#     filled=True,
#     rounded=True
# )
# plt.savefig('arvore_decisao.png', dpi=300, bbox_inches='tight')
# plt.show()
