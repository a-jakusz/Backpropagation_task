import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
import time
start = time.time()


#
# MNIST Data Loader Class (z kaggle.com)
#
class MnistDataloader(object):
    def __init__(self,
                 training_images_filepath = f"mnist_data/train-images.idx3-ubyte",
                 training_labels_filepath = f"mnist_data/train-labels.idx1-ubyte",
                 test_images_filepath = f"mnist_data/t10k-images.idx3-ubyte",
                 test_labels_filepath = f"mnist_data/t10k-labels.idx1-ubyte"
                 ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


loader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = loader.load_data() # odczytywanie danych
x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

np.random.seed()
# Number of hidden layers
K = 2
# Number of neurons per layer
D = 32
# Input layer
D_i = 28*28
# Output layer
D_o = 10

# Make empty lists
all_weights = [None] * (K+1)
all_biases = [None] * (K+1)

# Create input and output layers
all_weights[0] = np.random.normal(size=(D, D_i)) * np.sqrt(2 / D_i) # He initialization, overflow
all_weights[-1] = np.random.normal(size=(D_o, D)) * 0.01
all_biases[0] = np.ones((D,1)) * 0.01
all_biases[-1]= np.ones((D_o,1)) * 0.01

# Create intermediate layers
for layer in range(1,K):
  all_weights[layer] = np.random.normal(size=(D,D)) * np.sqrt(2 / D)
  all_biases[layer] = np.ones((D,1)) * 0.01

def a(preactivation): # funkcja aktywacji ReLU
    activation = preactivation.clip(0.0)
    return activation

def softmax(model_out):
    model_out = model_out - np.max(model_out) # dla stabilizacji, overflow czesto wystepuje
    exp_model_out = np.exp(model_out)
    sum_exp_model_out = np.sum(exp_model_out, axis = 0, keepdims=True) # axis = 0  -- sum along the column, keepdims -- like matlib part, dopasowanie pomiarow
    softmax_model_out = exp_model_out / sum_exp_model_out
    return softmax_model_out

def array_to_number(output_prob): # dla y prawidlowego
    return int(np.argmax(output_prob))

def number_to_array(number): # dla y podliczonego
    arr = np.zeros((10, 1))
    arr[number, 0] = 1
    return arr

def least_squares_loss(net_output, y): # loss_func
    net_output = np.array([o.flatten() for o in net_output]) # od 1D (60000,) do 2D (60000,10)
    y_arr = np.zeros((len(y), 10)) # 2D (60000, 10)
    for i in range(len(y)):
        y_arr[i] = number_to_array(y[i]).flatten() # od columny do wierszu
    return np.mean(np.sum((net_output-y_arr)**2, axis = 1)) # sumuje columny, otrzymujemy srednio per sample

def d_loss_d_output(net_output, y): # y -- prawidlowy, net_output -- podliczony
    y_arr = number_to_array(y)
    p = softmax(net_output)
    return 2*(p-y_arr)

def indicator_function(x):
    x_in = np.array(x)
    x_in[x_in>0] = 1
    x_in[x_in<=0] = 0
    return x_in

def backward_pass(all_weights, all_biases, all_f, all_h, y): # notebook 7.2, strona 7.4.1 w ksiazce
    all_dl_dweights = [None] * (K+1)
    all_dl_dbiases = [None] * (K+1)

    all_dl_df = [None] * (K+1)
    all_dl_dh = [None] * (K+1)

    all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y)) # 2*(y_out - y)

    for layer in range(K,-1,-1):
        all_dl_dbiases[layer] = all_dl_df[layer]
        all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].T)
        all_dl_dh[layer] = np.matmul(all_weights[layer].T, all_dl_df[layer])

        if layer > 0:
            all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * (np.array(np.matmul(all_weights[layer].T, all_dl_df[layer])))

    return all_dl_dweights, all_dl_dbiases


def training(all_weights, all_biases, x_train, y_train, l_rate = 0.001, epochs = 5):
    K = len(all_weights) - 1
    all_f = [None] * (K + 1)
    all_h = [None] * (K + 1)
    output = np.zeros(len(x_train), dtype=object) # z arrayami
    output_numb = np.zeros(len(x_train), dtype=int) # z liczbami

    for epoch in range(epochs):
        for i in range(len(x_train)):
            x = x_train[i].flatten() # from 28x28 to 784x1
            x = x / 255 # so that 0 -- black, 1 -- white
            x = x.reshape(-1, 1) # od wierszu do columny
            all_h[0] = x

            for layer in range(K):
                all_f[layer] = np.matmul(all_weights[layer], all_h[layer]) + all_biases[layer] # f = (omega * x) + beta
                all_h[layer+1] = a(all_f[layer]) # h = ReLU(f)

            all_f[K] = np.matmul(all_weights[-1], all_h[K]) + all_biases[-1] # dla ostatniego f
            output[i] = softmax(all_f[K]) # od 0 do 1 prawdopodobienstwo
            output_numb[i] = array_to_number(output[i])

            dl_dw, dl_db = backward_pass(all_weights, all_biases, all_f, all_h, y_train[i]) # backpropogation
            for j in range(K+1):
                all_weights[j] = all_weights[j] - l_rate * dl_dw[j] # w = w - nu * dl_dw
                all_biases[j] = all_biases[j] - l_rate * dl_db[j] # b = b - nu * dl_db

        loss = least_squares_loss(output, y_train)

        print("For epoch", epoch + 1)
        #print(output[120])
        print("Loss: ", loss)
    return output, all_weights, all_biases

output, all_weights, all_biases = training(all_weights, all_biases, x_train, y_train)
#print(y_train[120])

#plt.imshow(x_train[120], cmap='gray')
#plt.axis('off')
#plt.show()

def predictor(all_weights, all_biases, x_test):
    K = len(all_weights) - 1
    all_f = [None] * (K + 1)
    all_h = [None] * (K + 1)
    output = np.zeros(len(x_test), dtype=object)
    output_numb = np.zeros(len(x_test), dtype=int)

    for i in range(len(x_test)):
        x = x_test[i].flatten()
        x = x / 255
        x = x.reshape(-1, 1)
        all_h[0] = x

        for layer in range(K):
            all_f[layer] = np.matmul(all_weights[layer], all_h[layer]) + all_biases[layer]  # f = (omega * x) + beta
            all_h[layer + 1] = a(all_f[layer])  # h = ReLU(f)

        all_f[K] = np.matmul(all_weights[-1], all_h[K]) + all_biases[-1]  # dla ostatniego f
        output[i] = softmax(all_f[K])  # od 0 do 1 prawdopodobienstwo
        output_numb[i] = array_to_number(output[i])

    return output_numb

output_numb = predictor(all_weights, all_biases, x_test)

def checker(output, y_test):
    er = 0
    il = len(y_test)
    for i in range(len(y_test)):
        if output[i] != y_test[i]:
            er += 1
    res = 100*(il - er) / il
    return res

res = checker(output_numb, y_test)


end = time.time()
print(f"Czas wykonania: {end-start:.2f} s")
print(f"Wynik: {res:.2f}%")