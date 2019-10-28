#client side
from Crypto.Cipher import AES
from Crypto import Random
import base64
import socket
import serial
import sys
from preprocess import *
import threading
import time
from scipy import fft
import pickle


fileName = "Data.txt"
port = "/dev/ttyS0"
ser = serial.Serial(port, baudrate=115200)
f = open( fileName, "w+")

host = "192.168.43.111"
PORT_NUM = 7654

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#need to include ip addres of server to be connected
sock.connect((host, PORT_NUM))
bs = 32; #base_size
key = "1234567890123456"
x = 0

labels_dict = { 1: 'handmotor', 2: 'bunny', 3: 'tapshoulders', 4: 'rocket', 5: 'cowboy',
                6: 'hunchback', 7: 'jamesbond', 8: 'chicken', 9: 'movingsalute', 10: 'whip',
                0: 'idlemove', 11: 'exitmove'}

dataArray = []
dataList = []
powerReadingsArray = []

windowSize = 40
overlapPercentage = 0.3

def handshake():
    print("Sending Arduino \"Hello\"")
    ser.write('h'.encode())
    start = 0
    print("Sent hello")
    while start == 0: 
        print("in first loop")
        x = ser.read(1).decode('utf-8')
        print(x)
        if x == 'a':
            print("sending a")
            ser.write('a'.encode())
            start = 1
            print(start)
            print("sent ack back")
            
    while start == 1:
        print("here")
        if ser.in_waiting:
            if ser.read(1).decode('utf-8') == 'c':
                print("connected to arduino")
                start = 0
                
def readData():
    #receiving data from arduino
        ser.flush()
        if ser.in_waiting :
            reading = ser.read_until().decode("utf-8")
        if reading[0] == "#" :
            reading = reading.strip('#')
            print(reading)
            sensorReading = reading.split(':')[0]
            powerReading = reading.split(':')[1]
            sensor1Reading = sensorReading.split('|')[0]
            sensor2Reading = sensorReading.split('|')[1]
            sensor3Reading = sensorReading.split('|')[2]

            sensor1aX = float(sensor1Reading.split(',')[0])
            sensor1aY = float(sensor1Reading.split(',')[1])
            sensor1aZ = float(sensor1Reading.split(',')[2])
            sensor1gX = float(sensor1Reading.split(',')[3])
            sensor1gY = float(sensor1Reading.split(',')[4])
            sensor1gZ = float(sensor1Reading.split(',')[5])

            sensor2aX = float(sensor2Reading.split(',')[0])
            sensor2aY = float(sensor2Reading.split(',')[1])
            sensor2aZ = float(sensor2Reading.split(',')[2])
            sensor2gX = float(sensor2Reading.split(',')[3])
            sensor2gY = float(sensor2Reading.split(',')[4])
            sensor2gZ = float(sensor2Reading.split(',')[5])

            sensor3aX = float(sensor3Reading.split(',')[0])
            sensor3aY = float(sensor3Reading.split(',')[1])
            sensor3aZ = float(sensor3Reading.split(',')[2])
            sensor3gX = float(sensor3Reading.split(',')[3])
            sensor3gY = float(sensor3Reading.split(',')[4])
            sensor3gZ = float(sensor3Reading.split(',')[5])

            current = float(powerReading.split(',')[0])
            voltage = float(powerReading.split(',')[1])
            power = float(powerReading.split(',')[2])
            energy = float(powerReading.split(',')[3])

            powerReadingsArray = [current, voltage, power, energy]

            dataArray = [sensor1aX, sensor1aY, sensor1aZ, sensor1gX, sensor1gY, sensor1gZ,
                             sensor2aX, sensor2aY, sensor2aZ, sensor2gX, sensor2gY, sensor2gZ,
                             sensor3aX, sensor3aY, sensor3aZ, sensor3gX, sensor3gY, sensor3gZ]

            #print(dataArray)

            return dataArray
                
def encryptText(plainText, key):
    raw = pad(plainText)
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key.encode("utf8"),AES.MODE_CBC,iv)
    msg = iv + cipher.encrypt(raw.encode('utf8'))
    # msg = msg.strip()
    return base64.b64encode(msg)

def pad(var1):
    #var1 = var1 + (bs - (len(var1) % bs)) * ' '
    #var1 = var1.encode('utf-8') + (bs - (len(va
    #print("data size:" + str(len(var1)))
    #return var1
    return var1 + (bs - len(var1)%bs)*chr(bs - len(var1)%bs)

def sendToServer(action, shouldClose):
    powerReadingStr = "|" + "|".join(powerReadingsArray) + "\n"
    finalString = action + powerReadingStr
    
    finalString = encryptText(finalString,key)
    print(finalString)
    sock.send(finalString)
    #counter = counter+1
    if (shouldClose):
        sock.close()
        
class ArduinoToPiComms(threading.Thread):
    def _init_(self):
        threading.Thread._init_(self)
    def run(self):

        global dataList
        while (True):
            
            # Run your data collection from arduino here
            data = readData()
            data = np.asarray(data)
            if(data.shape[0] == 18):
                dataList.append(data)
            
                while len(dataList)>50:
                #check the systex for me might be wrong
                    dataList.pop(0)



class MLServer(threading.Thread):

    #List of Predictions
    predictedList = []
    #Number of consequitive predictions
    pNumber = 3

    def _init_(self):
        global model
        # Load the Model
        filename = "main_model.sav"
        model = pickle.load(open(filename, 'rb'))
        threading.Thread.__init__(self)


    def run(self):
        while (True):

            prediction = self.predict()

            if prediction == -1 :
                print("Invalid Prediction")
                time.sleep(0.5)
            elif prediction == 0:
                print("Prediction " + str(prediction))
                # Map value to dance move and send to server
                result_int = prediction
                danceMove = labels_dict[result_int]
                time.sleep(0.5)
            else:
                print("Prediction " + str(prediction))
                #Map value to dance move and send to server
                result_int = prediction
                danceMove = labels_dict[result_int]
                
                if ( danceMove == "exitmove" ):
                    sys.exit()
                
                sendToServer(danceMove, key)
                time.sleep(5)

        # Process the RawData

    def processdata(self):
        global dataList
        # Copy the Raw Data
        rawWindowData = np.asarray(dataList.copy())
        # Delete 1 - Overlap of data
        dataList = dataList[int(windowSize * (1 - overlapPercentage)):][:]

        processedData = []
        for sensorsIndex in range(rawWindowData.shape[1]):
            sensorData = rawWindowData[:][sensorsIndex]
            processedData.append(mean(sensorData))

            processedData.append(std(sensorData))
            processedData.append(mad(sensorData)[0])
            processedData.append(max(sensorData))
            processedData.append(min(sensorData))
            processedData.append(energy(sensorData))
            processedData.append(iqr(sensorData))

            sigma, rho = arCoeff(sensorData)

            processedData.append(sigma[0])
            processedData.append(sigma[1])
            processedData.append(sigma[2])
            processedData.append(sigma[3])
            processedData.append(rho)

            processedData.append(correlation(sensorData))
            fSensorData = fft(sensorData)

            processedData.append(maxInds(fSensorData))

            meanf = meanFreq(fSensorData)

            processedData.append(meanf.real + meanf.imag)

            skewnessf = skewness(fSensorData)

            processedData.append(skewnessf.real + skewnessf.imag)

            kurtosisf = kurtosis(fSensorData)

            processedData.append(kurtosisf.real + kurtosisf.imag)

        processedData.append(sma(rawWindowData[:][0], rawWindowData[:][1], rawWindowData[:][2]))
        processedData.append(sma(rawWindowData[:][3], rawWindowData[:][4], rawWindowData[:][5]))
        processedData.append(sma(rawWindowData[:][6], rawWindowData[:][7], rawWindowData[:][8]))
        processedData.append(sma(rawWindowData[:][9], rawWindowData[:][10], rawWindowData[:][11]))
        processedData.append(sma(rawWindowData[:][12], rawWindowData[:][13], rawWindowData[:][14]))
        processedData.append(sma(rawWindowData[:][15], rawWindowData[:][16], rawWindowData[:][17]))

        processedData = np.asarray(processedData)
        processedData[np.isnan(processedData)] = 0

        return processedData

    def predict(self):
        global model
        global dataList

        ## Check if number of data same as window size
        if (len(dataList)) == windowSize:

            print("Processing")
            processedData = self.processdata()

            test_prediction = model.predict([processedData])
            predictedValue = test_prediction[0]

            self.predictedList.append(predictedValue)

            ## if len of predicted List is more than number delete
            while len(self.predictedList) > self.pNumber:
                print(self.predictedList)
                del self.predictedList[0]
                print(self.predictedList)

            # check if len of predicted list of correct
            if len(self.predictedList) == self.pNumber:
                # check if all values in predicted list are the same
                if self.predictedList.count(self.predictedList[0]) == len(self.predictedList):
                    # delete the predicted list
                    valueToReturn = self.predictedList[0].copy()
                    del self.predictedList[:]
                    return valueToReturn
        return -1


print("Starting")
thread1 = ArduinoToPiComms()
thread2 = MLServer()
thread1.start()
thread2.start()

while(True):
    time.sleep(1)