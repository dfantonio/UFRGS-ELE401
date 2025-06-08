# from https://stackoverflow.com/questions/676172/full-examples-of-using-pyserial-package#7654527

import time
import joblib
import serial
import struct

x_test, y_test = joblib.load('breast_cancer_quantized.pkl')
y_test_numeric = [1 if diagnosis == 'M' else 0 for diagnosis in y_test]


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
tempo_execucao = []

# Display the first 2 lines of X_test_quantized
# print("First 2 rows of X_test_quantized:")
# print(x_test[:2])


# For a single float value
def float_to_bytes(value):
    return struct.pack('<f', value)  # Little-endian 32-bit float

# For your entire feature vector


def send_feature_vector(vector):
    # Pack all floats into a single byte string
    byte_data = struct.pack(f'<{len(vector)}f', *vector)
    return byte_data


def send_feature_vector_int(vector):
    # Pack integers into a single byte string (8-bit signed integers)
    byte_data = struct.pack(f'<{len(vector)}b', *vector)
    return byte_data


for i in range(0, tamanho_dataset):
    # for i in range(0, 2):
    # for i in range(0, 4):

    # Converte float para bytes
    # payload = send_feature_vector(x_test[i])

    # Converte inteiro para bytes
    payload = send_feature_vector_int(x_test[i])

    print(payload)

    ser.write(payload)

    if (i % 100 == 0):
        print(f'Enviando dados para o microcontrolador com o índice {i}')

    # print(f'Payload: {payload} {len(payload)}')

    # Aguarda receber um byte
    resposta = ser.read(10)  # Increase buffer size for longer response
    if b'ok' in resposta:
        parts = resposta.split(b':')

        valor_resposta = int(chr(parts[0][2]))
        tempo_exec = int(parts[1]) if len(parts) > 1 else 0
        respostas.append(valor_resposta)
        tempo_execucao.append(tempo_exec)
        # print(f"Sample {i}: Prediction={valor_resposta}, Time={tempo_exec} us")

        # exit()

    else:
        print(
            f'Erro ao receber linha {i}. Payload: {payload} Resposta: {resposta}')
        exit()


ser.close()

# Divide o tempo de execução por 84 pois o microcontrolador está rodando a 84MHz
tempo_execucao = [tempo / 84 for tempo in tempo_execucao]

print('\n\n Resultados:\n')


def check_accuracy(predictedY, Y):
    correctly_trained = 0
    total_train = 0

    # print(f'Gabarito | Microcontrolador')

    for i in range(0, len(predictedY)):
        # print(f'{Y[i]} | {predictedY[i]}')
        if (predictedY[i] == Y[i]):
            correctly_trained += 1
        total_train += 1

    percent_correctly_trained = correctly_trained/total_train * 100
    return percent_correctly_trained


accuracy = check_accuracy(respostas, y_test_numeric[:tamanho_dataset])
print(f'Acurácia: {accuracy}')

print(
    f'Tempo médio de execução: {round(sum(tempo_execucao)/len(tempo_execucao), 4)} us')
