import numpy as np
import math


def to_duel(other):
    if isinstance(other, Duel):
        return other
    return Duel(other, [0] * Duel.n_dim)

class Duel():
    n_dim = None
    def __init__(self, val, der):
        self.val = val
        self.der = np.array(der, dtype = float)

        if Duel.n_dim == None:
            Duel.n_dim = len(der)
        elif len(der) != Duel.n_dim:
            raise ValueError("derivative dimension mismatch")


    def __add__(self, other):
        other = to_duel(other)
        return Duel(self.val + other.val, self.der + other.der)

    def __radd__(self, other):
        other = to_duel(other)
        return Duel(self.val + other.val, self.der + other.der)

    def __sub__(self, other):
        other = to_duel(other)
        return Duel(self.val - other.val, self.der - other.der)

    def __rsub__(self, other):
        other = to_duel(other)
        return Duel(other.val - self.val, other.der - self.der)

    def __neg__(self):
        return Duel(-self.val, -self.der)

    def __mul__(self, other):
        other = to_duel(other)
        return Duel(self.val * other.val, self.val * other.der + other.val * self.der)

    def __rmul__(self, other):
        other = to_duel(other)
        return Duel(self.val * other.val, self.val * other.der + other.val * self.der)

    def __truediv__(self, other):
        other = to_duel(other)

        return Duel(self.val / other.val,
                    (self.der * other.val - self.val * other.der) / ((other.val)**2))

    def __rtruediv__(self, other):
        other = to_duel(other)

        return Duel(other.val / self.val,
                    (other.der * self.val - other.val * self.der) / ((self.val)**2))

    def __pow__(self, other):
        other = to_duel(other)
        return Duel(self.val ** other.val,
                    ((self.val ** other.val) * ((other.val * ((1/self.val) * self.der)) + (math.log(self.val) * other.der))))


def exp(var):
    var = to_duel(var)
    return Duel(math.exp(var.val), math.exp(var.val) * var.der)

def log(var):
    var = to_duel(var)
    return Duel(math.log(var.val), (1 / var.val) * var.der)
def sqrt(var):
    var = to_duel(var)
    return Duel(math.sqrt(var.val), (1 / (2 * math.sqrt(var.val))) * var.der)


def sin(var):
    var = to_duel(var)
    return Duel(math.sin(var.val), math.cos(var.val) * var.der)

def cos(var):
    var = to_duel(var)
    return Duel(math.cos(var.val), -math.sin(var.val) * var.der)

def tan(var):
    var = to_duel(var)
    return Duel(math.tan(var.val), (1 / (math.cos(var.val) ** 2)) * var.der)

def Relu(var):
    var = to_duel(var)
    val =0 if var.val < 0 else var.val
    der = 0 if var.val <= 0 else 1

    return Duel(val, der * var.der)






