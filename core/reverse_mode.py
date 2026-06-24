import math
import numpy as np

def to_node(data):
    if isinstance(data, Node):
        return data
    return Node(data)

def broadcast_correct(orginal, broadcasted):
    if orginal.shape != broadcasted.shape:
        changed_dim = np.where(np.array(orginal.shape) != np.array(broadcasted.shape))[0]
        return broadcasted.sum(axis = tuple(changed_dim), keepdims = True)
    return broadcasted

class Node:
    nodes = set() #storing all nodes, to make grad = 0 every time
    def __init__(self, val, der=0, backprop=None, parent=None):
        self.val = np.array(val, dtype= float)
        self.der = np.zeros_like(self.val, dtype = float)
        self.backprop = backprop
        self.parent = [] if parent == None else parent

        Node.nodes.add(self)

    def __add__(self, other):
        other = to_node(other)
        val = self.val + other.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, 1 * grad)
            other.der += broadcast_correct(other.der, 1 * grad)

        out.backprop = backprop
        return out

    def __radd__(self, other):
        other = to_node(other)
        val = self.val + other.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, 1 * grad)
            other.der += broadcast_correct(other.der, 1 * grad)

        out.backprop = backprop
        return out

    def __sub__(self, other):
        other = to_node(other)
        val = self.val - other.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, 1 * grad)
            other.der += broadcast_correct(other.der, -1 * grad)

        out.backprop = backprop
        return out

    def __rsub__(self, other):
        other = to_node(other)
        val = other.val - self.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, -1 * grad)
            other.der += broadcast_correct(other.der, 1 * grad)

        out.backprop = backprop
        return out

    def __truediv__(self, other):
        other = to_node(other)
        val = self.val / other.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, (1 / other.val) * grad)
            other.der += broadcast_correct(other.der, (-self.val / ((other.val) ** 2)) * grad)

        out.backprop = backprop
        return out

    def __rtruediv__(self, other):
        other = to_node(other)
        val = other.val / self.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, (-other.val / ((self.val) ** 2)) * grad)
            other.der += broadcast_correct(other.der, (1 / self.val) * grad)

        out.backprop = backprop
        return out

    def __mul__(self, other):
        other = to_node(other)
        val = self.val * other.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, grad * other.val)
            other.der += broadcast_correct(other.der, self.val * grad)

        out.backprop = backprop
        return out

    def __rmul__(self, other):
        other = to_node(other)
        val = self.val * other.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, other.val * grad)
            other.der += broadcast_correct(other.der, self.val * grad)

        out.backprop = backprop
        return out

    def __pow__(self, other):
        other = to_node(other)
        val = self.val ** other.val
        out = Node(val, parent=[self, other])

        def backprop(grad):
            self.der += broadcast_correct(self.der, (other.val * (self.val ** (other.val - 1))) * grad)
            other.der += broadcast_correct(other.der, ((self.val ** other.val) * math.log(self.val)) * grad)

        out.backprop = backprop
        return out

def matmul(node1, node2):
    node1 = to_node(node1)
    node2 = to_node(node2)
    out = Node(node1.val @ node2.val, parent = [node1, node2])

    def backprop(grad):
        node1.der += broadcast_correct(node1.der, grad @ node2.val.T)
        node2.der += broadcast_correct(node2.der, node1.val.T @ grad)
    out.backprop = backprop
    return out

def sin(x):
    x = to_node(x)
    out = Node(np.sin(x.val), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, np.cos(x.val) * grad)

    out.backprop = backprop
    return out

def cos(x):
    x = to_node(x)
    out = Node(np.cos(x.val), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, -np.sin(x.val) * grad)

    out.backprop = backprop
    return out

def tan(x):
    x = to_node(x)
    out = Node(np.tan(x.val), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, (1 / (np.cos(x.val) ** 2)) * grad)

    out.backprop = backprop
    return out


def log(x):
    x = to_node(x)
    out = Node(np.log(x.val), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, (1 / x.val) * grad)

    out.backprop = backprop
    return out

def sqrt(x):
    x = to_node(x)
    out = Node(np.sqrt(x.val), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, (1 / (2 * np.sqrt(x.val))) * grad)

    out.backprop = backprop
    return out


def relu(x):
    x = to_node(x)
    out = Node(np.where(x.val > 0, x.val, 0), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, np.where(x.val <= 0, 0, 1) * grad)

    out.backprop = backprop
    return out


def sigmoid(x):
    x = to_node(x)
    out = Node(1 / (1 + np.exp(-x.val)), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, (out.val * (1 - out.val)) * grad)

    out.backprop = backprop
    return out

def tanh(x):
    x = to_node(x)
    out = Node((np.exp(x.val) - np.exp(-x.val)) / (np.exp(x.val) + np.exp(-x.val)), parent=[x])

    def backprop(grad):
        x.der += broadcast_correct(x.der, (1 - (out.val ** 2)) * grad)

    out.backprop = backprop
    return out

def mse_loss(x, target):
    x = to_node(x)
    x_val = x.val.flatten() #flattening so it be easy to subtract from true value and find error
    target = np.array(target).flatten()
    out = Node( np.mean((target - x_val)**2), parent = [x] )

    def backprop(grad):
        x.der += broadcast_correct(x.der,  np.mean(2*(target - x_val)) * grad)
    out.backprop = backprop
    return out

def cross_entropy_loss(x, target):
    # softmax
    x = to_node(x)
    p = np.exp(x.val)
    p = p / np.sum(p, axis=0, keepdims=True)

    #cross entropy
    one_hot = np.zeros(x.val.shape)
    one_hot[target, range(x.val.shape[1])] = 1

    loss = -np.mean( np.log( p[target, range(x.val.shape[1])] ) )
    out = Node(loss, parent = [x])

    def backprop(grad):
        # dividing with no.of sample, so we know each neuron/class contribution in loss, like that for all samples, helps to not explode derivative
        x.der += broadcast_correct(x.der, ((p - one_hot)/ x.val.shape[1]) * grad)
    out.backprop = backprop
    return out

def zero_grad():
    copy_nodes = Node.nodes.copy()
    for i in copy_nodes:
        if not i.parent:# if no parents(w,b,x) nodes remainig dependent nodes removed
            i.der = i.der * 0
        else:
            Node.nodes.remove(i)


def topo(node, seen, nodes):
    if node not in seen:
        seen.add(node)
        for i in node.parent:
            topo(i, seen, nodes)
        nodes.append(node)

def backprop(out):
    out.der = np.ones_like(out.val, dtype = float)
    seen = set()
    nodes = []

    topo(out, seen, nodes)

    for i in reversed(nodes):
        if i.backprop:
            i.backprop(i.der)


