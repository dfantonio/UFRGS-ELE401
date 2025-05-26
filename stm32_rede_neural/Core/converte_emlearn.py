import joblib
import emlearn

# Carregar o modelo treinado
modelo = joblib.load('modelo_rede_neural.joblib')

# Converter para C
cmodel = emlearn.convert(modelo)
cmodel.save(file='./Inc/rede_neural.h')
