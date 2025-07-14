import joblib
import emlearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


x_test = joblib.load('x_test.joblib')
y_test = joblib.load('y_test.joblib')

# Carregar o modelo treinado
modelo = joblib.load('modelo_rede_neural.joblib')

# Converter para C
cmodel = emlearn.convert(modelo)
cmodel.save(file='./Inc/rede_neural.h')

predictedY = modelo.predict(x_test)
accuracy = accuracy_score(y_test, predictedY)
print(f'Acurácia CONVERTE: {accuracy}')
