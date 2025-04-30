from sklearn import tree
import numpy as np
# import ipdb

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

feature_names = ['x1', 'x2']
target_names = ['falso', 'verdadeiro']

clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X, Y)


# export in text format
r = tree.export_text(clf)
print('\n'+r)


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


bits = 3
max_value = 2**bits
clf_qt = my_quantized_DT(clf, bits)

X_temp = X*max_value
X_temp = np.floor(X_temp)
X_qt = X_temp.astype('int')

Y_qt = clf_qt.predict(X_qt)
print(Y_qt)
