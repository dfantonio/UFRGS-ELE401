import joblib
import emlearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dataset = datasets.load_digits()

RANDOM_STATE = 2  # ou qualquer outro número

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

# Carregar o modelo treinado
modelo = joblib.load('modelo_rede_neural.joblib')

# Converter para C
cmodel = emlearn.convert(modelo)
cmodel.save(file='./Inc/rede_neural.h')

predictedY = modelo.predict(X_val)
accuracy = accuracy_score(y_val, predictedY)
print(accuracy)
