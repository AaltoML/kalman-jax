import pickle
import numpy as np

method_nlpd = np.zeros([17, 10])
for method in range(17):
    for fold in range(10):
        with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
            method_nlpd[method, fold] = pickle.load(fp)

print(np.mean(method_nlpd, axis=1))
print(np.std(method_nlpd, axis=1))
