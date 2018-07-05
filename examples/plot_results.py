# after running
# python train-pytorch-simple.py | tee results.txt
# this script will plot the validation accuracy

import numpy as np
import matplotlib.pyplot as plt
import re

re_pat = 'Test set: Average loss: ([^,]+), Accuracy: ([^\%]+)'
acc = []

with open('results.txt') as f:
    for line in f:
        if 'Test set:' in line:
            print(line[:-1])
            loss_acc = re.search(re_pat, line[:-1])
            acc.append(float(loss_acc.group(2)))

acc = np.asarray(acc)
x = [i*50000/1e6 for i in range(1, len(acc))]
y = [np.amax(acc[:i]) for i in range(1, len(acc) + 1)]
plt.plot(x, y)
plt.xlabel('cumulative samples seen in training (M)')
plt.ylabel('validation accuracy')
plt.title('max validation accuracy')
plt.grid(True)
plt.savefig('training-4-4.png')
plt.show()
