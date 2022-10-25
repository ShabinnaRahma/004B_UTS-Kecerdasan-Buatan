#Shabinna Rahmadilla Santoso_21091397004
#Multiple perceptron / Neuron batch and multiple layer 2

#inisialisasi numpy
import numpy as np

# inisialisasi variabel
# memasukan nilai variabel layer feature 10 dengan batch sejumlah 6 
inputs = [[2.5, 1.9, 7.2, 0.6, 1.3, 2.9, 1.4, 0.9, 2.4, 3.9],
		  [4.7, 1.2, 1.14, 5.5, 2.5, 3.2, 0.5, 1.9, 1.7, 1.0],
		  [3.7, 0.17, 4.3, 2.12, 0.11, 0.8, 2.1, 3.8, 1.5, 2.8],
		  [2.4, 0.20, 5.7, 3.3, 1.4, 1.3, 0.2, 0.10, 2.8, 1.4],
		  [0.11, 4.5, 3.0, 0.14, 2.9, 2.9, 1.8, 3.5, 5.3, 1.2],
		  [1.0, 1.3, 2.4, 0.15, 3.0, 3.6, 4.0, 4.2, 4.8, 5.7]]

# memberikan nilai bobot pada variabel sesuai dengan jumlah input
# memasukan jumlah weight sesuai dengan jumlah neuron yaitu sejumlah 5  
weights1 = [[-2.1, 3.0, 2.2, 3.8, -1.0, 0.8, -0.3, 2.4, -0.6, 1.3],
		   [4.8, 4.5, -0.7, 1.1, 4.9, 3.5, 0.4, -2.2, 1.0, 4.5],
		   [-0.1,-0.4, 1.3, 0.9, 0.2, 1.5, -1.7, 2.2, 2.8, 2.1],
		   [1.4,-0.9, 0.2, -0.10, 5.5, 3.3, -1.3, 4.9, 2.5, -3.7],
		   [3.0, 4.1,-1.3, 1.5, 1.6, -2.7, 2.9, 2.2, -0.3, -1.2]]

# inisialisasi biases pada layer1 sesuai dengan neuron yang ditentukan yaitu layer 1 = 5 neuron
biases1 =   [0.3, 1.9, 2.3, 3.5, 1.7]

# inisialisasi jumlah weight 2, weight layer 2 = neuron layer 1 yaitu 5
# memasukkan jumlah weight sesuai dengan neuron layer 2 yaitu 3 neuron
weights2 = [[-0.2, 1.5, 1.1, 2.1, -1.3],
			[0.4, 1.6, 2.4, 3.0, 2.2],
			[2.4, 2.5, 3.0, 1.1, 1.5]]

# inisialisasi biases pada layer2 dengan neuron yang ditentukan yaitu 3			
biases2 =  [2.0, 1.0, 0.4]

# output
# menghitung layer1 dengan (inputs*weight1) dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# menghitung layer2 dengan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print output layer2
print(layer2_outputs) 