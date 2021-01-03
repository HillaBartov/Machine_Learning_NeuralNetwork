import sys
import numpy as np


def sigmoid(z1):
    return 1 / (1 + np.exp(-z1))


def softmax(z2):
    z2 = z2 - np.max(z2)
    return np.exp(z2) / np.sum(np.exp(z2))


def sigmoid_derivative(h1):
    return h1 * (1 - h1)


def forward(params, x):
    # Forward Propagation
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1, x) + b1
    # hidden layer
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    # out layer
    h2 = softmax(z2)
    loss = -np.log(h2)
    return {'x': x, 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2,
            'loss': loss}


def back(cache):
    # Back Propagation
    x, w2, z1, h1, z2, h2, loss, y = [cache[key] for key in ('x', 'w2', 'z1', 'h1', 'z2', 'h2', 'loss', 'y')]
    # softmax_derivative
    h2[int(y)][0] -= 1
    dz2 = h2
    dw2 = np.dot(dz2, h1.T)
    db2 = dz2
    dz1 = np.dot(w2.T, dz2) * sigmoid_derivative(h1)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1
    # update
    return {'w1': dw1, 'b1': db1, 'w2': dw2, 'b2': db2}


def train():
    w1 = np.random.uniform(-1, 1, size=(100, 784)) * 0.01
    b1 = np.zeros((100, 1))
    w2 = np.random.uniform(-1, 1, size=(10, 100)) * 0.01
    b2 = np.zeros((10, 1))
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    learning_rate = 0.01
    epochs = 20
    for e in range(epochs):
        # shuffle train data every epoch
        s = np.arange(train_x.shape[0])
        np.random.shuffle(s)
        train_x_shuffled = train_x[s]
        train_y_shuffled = train_y[s]
        for x_i, y_i in zip(train_x_shuffled, train_y_shuffled):
            x_i = x_i.reshape(784, 1)
            cache = forward(params, x_i)
            cache['y'] = y_i
            gradients = back(cache)
            # update
            w1 = w1 - learning_rate * gradients['w1']
            w2 = w2 - learning_rate * gradients['w2']
            b1 = b1 - learning_rate * gradients['b1']
            b2 = b2 - learning_rate * gradients['b2']
            params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return params


def predict():
    test_y = []
    for x_i in test_x:
        x_i = x_i.reshape(784, 1)
        cache = forward(parameters, x_i)
        test_y.append(np.argmax(cache['h2']))
    return test_y


training_examples, training_labels, testing_examples = sys.argv[1], sys.argv[2], sys.argv[3]
train_x = np.loadtxt(training_examples)
train_y = np.loadtxt(training_labels)
test_x = np.loadtxt(testing_examples)
train_size = len(train_x)
test_size = len(test_x)
# normalize data
train_x /= 255
test_x /= 255
parameters = train()
np.savetxt('test_y', np.array(predict()), fmt='%i')
