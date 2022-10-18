#Shabinna Rahmadilla Santoso_21091397004
#Multi neuron menggunakan numpy

#inisialisasi numpy
import numpy as np

#inisialisasi variabel
#memasukkan nilai variabel layer feature 10
inputs = [5.0, 9.0, 7.0, 11.0, 1.0, 4.0, 8.0, 12.0, 6.0, 10.0]

# memberikan nilai bobot pada variabel sesuai dengan jumlah input
# memasukan jumlah weight sesuai dengan jumlah neuron
weights = [[0.7, 0.9, 0.5, 0.11, 0.9, 0.3, 0.2, 0.3, 0.16, 0.4],
[0.12, 0.20, 0.3, 0.9, 0.19, 0.1, 0.4, 0.22, 0.24, 0.30],
[5.0, 2.7, 1.9, 0.13, 1.5, 0.20, 2.2, 0.36, 1.3, 2.9],
[1.0, 0.6, 0.13, 2.5, 0.6, 7.0, 2.4, 2.0, 0.35, 0.22],
[9.0, 0.9, 0.4, 0.2, 1.1, 0.45, 0.11, 0.34, 0.36, 1.1]]

#inisialisasi bias sesuai dengan neuron yang ditentukan
biases = [8.0, 6.0, 3.0, 1.0, 5.0]

#output
layer_outputs = np.dot(weights, inputs) + biases

#print output
print(layer_outputs)