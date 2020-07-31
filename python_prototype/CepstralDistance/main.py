import numpy as np
from scipy.io.wavfile import read
#Main script to make simple tests
from realceps import realceps
import matplotlib.pyplot as plt
from cepsdist import cepsdist

 #print(data)
data = [4,3,5,7]
data2 = [1,2,3,4,5,6,7,8,9,10]
x=[4,3,5,7],[2,3,4,5],[2,5,6,7],[1,2,3,4]

#output = realceps(data2)
#print(data2)
#plt.figure()
#plt.plot(output)
#plt.show()
cepsdist(data, data2)
