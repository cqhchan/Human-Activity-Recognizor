import numpy as np
import sys
import re
import os

def addDataFromFile(windowData, windowLabel, path, filenameInput):
    filename = os.path.join(path, file)
    f = open(filename, "r")
    data = []
    label = int(filenameInput[0])
    timePerData = 50/1000 # 50 Milliseconds
    windowDuration = 2 # 3 seconds
    dataPerWindow = int (windowDuration / timePerData)
    overlap = 0.3;
    for dataString in f:

      #  str = "0.41,-0.04,0.18,-22.39,-17.73,-13.47|0.08,0.11,1.00,13.48,19.89,28.49|0.08,-0.11,
        # -0.17,-86.15,28.58,69.42"
        values = [float(x.strip()) for x in re.split('\||,', dataString)]
        data.append(values)

    totalDataLength = len(data)
    windowRange = totalDataLength - dataPerWindow

    data = np.asarray(data)

    currentIndex = 0;
    while currentIndex <= windowRange:
        windowData.append(data[currentIndex: currentIndex + dataPerWindow][:])
        windowLabel.append(label)
        currentIndex += int(overlap * dataPerWindow)


path = sys.argv[1]
destination = sys.argv[2]
print(destination)
windowData = []
windowLabel = []



for file in os.listdir(path):
    if file.endswith(".txt"):
        print(os.path.join(path, file))
        addDataFromFile(windowData, windowLabel, path, file)

windowData = np.asarray(windowData)
print(windowData.shape)


windowLabel = np.asarray(windowLabel)

print(windowLabel.shape)


np.save(os.path.join(destination, "RawWindowedData"), windowData)
np.save(os.path.join(destination, "WindowedLabels"), windowLabel)
np.savetxt(os.path.join(destination, "WindowedLabelsReadable.txt"), windowLabel.astype(int))
