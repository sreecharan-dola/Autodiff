import numpy as np
from core import reverse_mode as rev

# z1 = w*x + b = (5, 2) X  (2 , 2) + (5, 1) = (5, 2)
# w1 * z1 + b1 = (2, 5) X (5, 2) + (2, 1) = (2, 2)


def train(x, true_index , hidden_neuron, classes, lr = 0.04, epoch = 700):

    x = rev.Node(x)
    w1 = rev.Node(np.random.randn(hidden_neuron, x.val.shape[0]))
    b1 = rev.Node(np.random.randn(hidden_neuron, 1))
    w2 = rev.Node(np.random.randn(classes, hidden_neuron))
    b2 = rev.Node(np.random.randn(classes, 1))

    for i in range(epoch):
        z1 = rev.matmul(w1, x) + b1
        a = rev.relu(z1)
        z2 = rev.matmul(w2, a) + b2
        out = rev.cross_entropy_loss(z2, true_index)

        rev.backprop(out)
        if i == 0:
            print('Initial loss - ', out.val)
        if i == epoch-1:
            print('Final loss - ', out.val)


        w1.val -= lr * w1.der
        b1.val -= lr * b1.der

        w2.val -= lr * w2.der
        b2.val -= lr * b2.der

        rev.zero_grad()
    return [w1,w2,b1,b2]

def testing(x, learned_vars):
    w1, w2, b1, b2 = learned_vars
    z1 = rev.matmul(w1, x) + b1
    a = rev.relu(z1)
    z2 = rev.matmul(w2, a) + b2

    # softmax
    p = np.exp(z2.val)
    p = p / np.sum(p, axis=0, keepdims=True)
    true_class_each_samp = np.argmax(p, axis=0)

    data = {'probability': p, 'true_class_for_each_sample':true_class_each_samp}
    return data

