# from https://stackoverflow.com/questions/676172/full-examples-of-using-pyserial-package#7654527

import time
import serial
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

dataset = datasets.load_digits()


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='COM5',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=5
    # parity=serial.PARITY_EVEN
    #     stopbits=serial.STOPBITS_TWO,
    #     bytesize=serial.SEVENBITS
)

ser.isOpen()


# Treina com o melhor número de neurônios
clf = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(195),
    activation='relu',
    max_iter=1000,
    random_state=2,
    alpha=0.0001,
    learning_rate='constant',
    learning_rate_init=0.001,
)

print('Iniciando envio de dados para o microcontrolador\n')

# for i in range(0, len(dataset.data)):

# print("Resposta verdadeira | Modelo Python | Microcontrolador")
# predictedY = clf.predict(dataset.data[:10])

tamanho_dataset = len(dataset.data)
# tamanho_dataset = 20
respostas = []

for i in range(0, tamanho_dataset):
    # print(
    #     f'Enviando dados para o microcontrolador com o índice {i}', dataset.data[i])
    payload = b''

    # for j in range(0, 4):
    for j in range(0, len(dataset.data[i])):
        valor = int(dataset.data[i][j])
        # Converte o inteiro para 1 byte
        byte_valor = valor.to_bytes(1, byteorder='little')
        payload += byte_valor

    # payload += b'\xff'
    ser.write(payload)
    if (i % 10 == 0):
        print(f'Enviando dados para o microcontrolador com o índice {i}')

    # print(f'Payload: {payload} {len(payload)}')

    # Aguarda receber um byte
    resposta = ser.read(3)  # Lê 2 bytes
    valor_resposta = int(chr(resposta[2]))
    if b'ok' in resposta:
        respostas.append(valor_resposta)
    else:
        print(
            f'Erro ao receber linha {i}. Payload: {payload} Resposta: {resposta}')
        exit()


def check_accuracy(predictedY, Y):
    correctly_trained = 0
    total_train = 0

    print(f'Gabarito | Microcontrolador')

    for i in range(0, len(Y)):
        print(f'{Y[i]} | {predictedY[i]}')
        if (predictedY[i] == Y[i]):
            correctly_trained += 1
        total_train += 1

    percent_correctly_trained = correctly_trained/total_train * 100
    return percent_correctly_trained


accuracy = check_accuracy(respostas, dataset.target[:tamanho_dataset])
print(f'Acurácia: {accuracy}')
