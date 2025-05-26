# from https://stackoverflow.com/questions/676172/full-examples-of-using-pyserial-package#7654527

import time
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='COM5',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
    #     parity=serial.PARITY_EVEN
    #     stopbits=serial.STOPBITS_TWO,
    #     bytesize=serial.SEVENBITS
)

ser.isOpen()

print('Enter your commands below.\r\nInsert "exit" to leave the application.')

varinput = 1
while 1:
    # varinput = input(">> ")

    # send the character to the device
    # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
    # serialcmd = varinput + '\r\n'
    # serialcmd = varinput  # + '\r'
    # ser.write(serialcmd.encode())
    out = ''
    # let's wait one second before reading output (let's give device time to answer)
    time.sleep(0.1)
    while ser.inWaiting() > 0:
        out += ser.read_all().decode()

    if out != '':
        print(">>" + out)
        out = ''
