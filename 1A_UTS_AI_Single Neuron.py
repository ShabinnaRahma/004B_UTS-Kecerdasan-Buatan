#Shabinna Rahmadilla Santoso_21091397004
#single neuron menggunakan numpy

#inisialisasi numpy
import numpy as np

# inisialisai variabel
# memasukan nilai variabel layer feature 10
inputs = [5, 9, 10, 2, 15, 6, 1, 8, 11, 18]

# memberikan nilai bobot pada variabel sesuai dengan jumlah input
weights = [0.3, 0.7, 0.9, 0.2, 0.4, 0.2, 0.1, 0.8, 0.5, -0.4]

#inisialisasi bias
bias = 8

#output
output = np.dot(weights, inputs) + bias

#print output
print(output)