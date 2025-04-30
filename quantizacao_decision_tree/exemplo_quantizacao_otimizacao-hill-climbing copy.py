from sklearn import tree
import numpy as np
import random

X = np.random.random([500, 2])
Y = []
for x_value in X:
    y_value = 0
    if (x_value[0] > 0.5) and (x_value[1] > 0.5):  # AND "logic"
        y_value = 1

    # random part
    if np.random.random() > 0.9:
        y_value = not y_value

    Y.append(y_value)

Y = np.array(Y)

# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# Y = np.array([0,1,1,0])

feature_names = ['x1', 'x2']
target_names = ['falso', 'verdadeiro']

clf = tree.DecisionTreeClassifier(criterion='gini')
# clf = tree.DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0.01)

clf = clf.fit(X, Y)
Y_original_predicted = clf.predict(X)

# export in text format
r = tree.export_text(clf)
print('\n'+r)


class my_quantized_DT():
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
                'int')   # value of comparison
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

    def print(self):

        # exploration in depth first
        depth = 1
        visited_nodes = []
        node = 0
        parent = 0

        while (node >= 0):
            if len(visited_nodes) > 0:
                times_node_visited = visited_nodes.count(node)
            else:
                times_node_visited = 0

            if times_node_visited == 0:
                th_operation = "<="
            elif times_node_visited == 1:
                th_operation = ">"  # second time in a node will be > operation
            else:
                break

            print("|   "*(depth-1), end='')
            print("|---", end='')
            if (self.feature[node] >= 0):
                print(" feature_%d %s %d" %
                      (self.feature[node], th_operation, self.threshold[node]))
            else:
                print(" class: %d" % (np.argmax(self.value[node])))

            visited_nodes.append(node)

            if ((self.children_left[node] not in visited_nodes) and (self.children_left[node] > 0)):
                next_node = self.children_left[node]
                parent = node
                depth += 1

            elif ((self.children_right[node] not in visited_nodes) and (self.children_right[node] > 0)):
                next_node = self.children_right[node]
                parent = node
                depth += 1
            else:
                next_node = parent
                depth -= 1

            node = next_node

        print("")

    def get_threshold(self):
        return self.threshold


bits = 4
max_value = 2**bits
clf_qt = my_quantized_DT(clf, bits)

X_temp = X*max_value
X_temp = np.floor(X_temp)
X_qt = X_temp.astype('int')

Y_qt = np.array(clf_qt.predict(X_qt))

clf_qt.print()

acuracy_between_original_and_quantized_dt = np.sum(
    Y_qt == Y_original_predicted)/len(Y_original_predicted)
print("Number of bits: %d" % bits)
print("Accuracy between original and quantized DT: %f" %
      acuracy_between_original_and_quantized_dt)


# Hill Climbing
def hill_climbing(x0, loss):
    x = x0
    cost = loss(x)
    iter = 0
    iter_without_gain = 0
    while (iter_without_gain < 100):
        # x_candidate = modificar(x)
        index = random.randint(0, len(x)-1)
        x_candidate = np.copy(x)
        x_candidate[index] += 1 if random.random() < 0.5 else -1
        cost_candidate = loss(x_candidate)
        if (cost_candidate < cost):
            cost = cost_candidate
            x = x_candidate
            iter_without_gain = 0
        else:
            iter_without_gain += 1
        # print("Iter: %d \t x_candidate: %s \t cost_candidate: %g cost: %g" %(iter, x_candidate, cost_candidate, cost))
    return x, cost


def loss_quantized_dt(x):
    clf_qt_attempt = my_quantized_DT(clf, bits, x)
    Y_qt_attempt = np.array(clf_qt_attempt.predict(X_qt))

    accuracy = np.sum(Y_qt_attempt == Y_original_predicted) / \
        len(Y_original_predicted)

    return -accuracy


original_thresholds = clf_qt.get_threshold()

x0 = original_thresholds

final_x, final_cost = hill_climbing(x0, loss_quantized_dt)

print("Final x: %s \t Final cost: %g" % (final_x, final_cost))
