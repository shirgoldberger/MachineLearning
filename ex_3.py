import sys
import numpy as np
from scipy.special import softmax

EPOCHS = 30
HIDDEN_LAYER = 256
SIZE_INPUT = 784
ETA = 0.01
LABELS = 10


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def load_and_normalize(train_x_file, train_y_file, test_x_file):
    # load
    train_x = np.loadtxt(train_x_file)
    train_y = np.loadtxt(train_y_file)
    test_x = np.loadtxt(test_x_file)
    # normalize
    train_x = np.divide(train_x, 255)
    test_x = np.divide(test_x, 255)
    return train_x, train_y, test_x


def fprop(x, w1, b1, w2, b2):
    x = np.array(x)
    x.shape = (784, 1)
    np.transpose(b1)
    np.transpose(b2)
    z1 = np.add(np.dot(w1, x), b1)
    h1 = sigmoid(z1)
    z2 = np.add(np.dot(w2, h1), b2)
    h2 = softmax(z2)
    return z1, h1, z2, h2


def backprop(x, y, z1, h1, h2, w2):
    vec_y = np.zeros((10, 1))
    vec_y[int(y)] = 1
    dz2 = (h2 - vec_y)  # dl/dz2
    dw2 = np.dot(dz2, h1.T)  # dl/dw2
    db2 = dz2  # dl/db2
    dh1 = np.dot(w2.T, dz2)  # dl/dh1
    dz1 = dh1 * sigmoid_derivative(z1)  # dl/dz1
    dw1 = np.dot(dz1, x.T.reshape(1, 784))  # dl/dw1
    db1 = dz1  # dl/db1
    return dw1, db1,  dw2, db2


def update_parameters(parameters, gradients):
    w1, b1, w2, b2 = parameters
    g_w1, g_b1, g_w2, g_b2 = gradients
    w1 -= (ETA * g_w1)
    b1 -= (ETA * g_b1)
    w2 -= (ETA * g_w2)
    b2 -= (ETA * g_b2)
    return w1, b1, w2, b2


def train(train_x, train_y, w1, b1, w2, b2):
    for i in range(EPOCHS):
        list_zip = list(zip(train_x, train_y))
        np.random.shuffle(list_zip)
        for x, y in list_zip:
            z1, h1, z2, h2 = fprop(x, w1, b1, w2, b2)
            g_w1, g_b1, g_w2, g_b2 = backprop(x, y, z1, h1, h2, w2)
            gradients = [g_w1, g_b1, g_w2, g_b2]
            parameters = [w1, b1, w2, b2]
            w1, b1, w2, b2 = update_parameters(parameters, gradients)
    return w1, b1, w2, b2


def main():
    train_x, train_y, test_x = load_and_normalize(sys.argv[1], sys.argv[2], sys.argv[3])

    w1 = np.random.uniform(-0.08, 0.08, [HIDDEN_LAYER, SIZE_INPUT])
    b1 = np.random.uniform(-0.08, 0.08, [HIDDEN_LAYER, 1])
    w2 = np.random.uniform(-0.08, 0.08, [LABELS, HIDDEN_LAYER])
    b2 = np.random.uniform(-0.08, 0.08, [LABELS, 1])
    w1, b1, w2, b2 = train(train_x, train_y, w1, b1, w2, b2)

    output_file = open("test_y", 'w')
    for x in test_x:
        x = np.reshape(x, (1, SIZE_INPUT))
        z1, h1, z2, y_hat = fprop(x, w1, b1, w2, b2)
        y_hat = y_hat.argmax(axis=0)
        output_file.write(str(y_hat[0]) + "\n")
    output_file.close()


if __name__ == '__main__':
    main()
