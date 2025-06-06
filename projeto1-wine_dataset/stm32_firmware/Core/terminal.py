# from https://stackoverflow.com/questions/676172/full-examples-of-using-pyserial-package#7654527

import time
import joblib
import serial
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

# dataset = datasets.load_digits()

x_test, y_test = joblib.load('breast_cancer.pkl')


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='COM5',
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=5
    # parity=serial.PARITY_EVEN
    #     stopbits=serial.STOPBITS_TWO,
    #     bytesize=serial.SEVENBITS
)

ser.isOpen()


tamanho_dataset = len(x_test)
print(
    f'Iniciando envio de dados para o microcontrolador com {tamanho_dataset} dados\n')

# for i in range(0, len(dataset.data)):

respostas = []

# Display the first 2 lines of X_test_quantized
print("First 2 rows of X_test_quantized:")
print(x_test[:2])

# Display the first 2 lines of y_test
print("\nFirst 2 rows of y_test:")
print(y_test[:2])


# for i in range(0, tamanho_dataset):
for i in range(0, 1):
    # print(
    #     f'Enviando dados para o microcontrolador com o índice {i}', dataset.data[i])
    payload = b''

    # for j in range(0, 4):
    for j in range(0, len(x_test[i])):
        valor = int(x_test[i][j])
        # Converte o inteiro para 1 byte
        byte_valor = valor.to_bytes(1, byteorder='little')
        payload += byte_valor

    # payload += b'\xff'
    ser.write(payload)
    # if (i % 100 == 0):
    print(f'Enviando dados para o microcontrolador com o índice {i}')

    print(f'Payload: {payload} {len(payload)}')

    # Aguarda receber um byte
    resposta = ser.read(3)  # Lê 2 bytes
    if b'ok' in resposta:
        valor_resposta = int(chr(resposta[2]))
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


# accuracy = check_accuracy(respostas, y_test[:tamanho_dataset])
# print(f'Acurácia: {accuracy}')
