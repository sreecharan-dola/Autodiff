import numpy as np
from core import forward_mode as fm
from nn import neural_network as nn

#forward mode
x = fm.Duel(2, [1, 0])
y = fm.Duel(3, [0, 1])
a = x * fm.sin(y)
b = fm.log(x)
f = a + b

for s,i in [('x',x), ('y',y), ('a',a), ('b',b), ('f',f)]:
    print(s + '- val :',i.val,  s + '- der :', i.der)


print('-----')

#reverse mode
classes = 2
hidden_neuron = 5
x = np.array([[1, 2, 3, 4, 5, 6, 1, 2],
                       [1, 1, 2, 2, 3, 3, 4, 5]])
true_index = [0, 0, 0, 1, 1, 1, 0, 1]

learnings = nn.train(x, true_index, hidden_neuron, classes)
x_test = np.array([[1, 4, 3, 6],
              [2, 1, 4, 2]])
output = nn.testing(x_test,learnings)

print(output)
