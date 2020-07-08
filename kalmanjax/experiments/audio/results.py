import pickle
import numpy as np

method_nlpd = np.zeros([17, 10])
for method in [0, 1, 2, 3, 4, 5, 9, 10, 11, 15]:
    for fold in range(10):
        with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
            method_nlpd[method, fold] = pickle.load(fp)

# for fold in range(10):
#     with open("output/" + str(15) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#         print(pickle.load(fp))

np.set_printoptions(precision=3)
print(np.mean(method_nlpd, axis=1))
# print(np.nanmean(method_nlpd, axis=1))
np.set_printoptions(precision=2)
print(np.std(method_nlpd, axis=1))
# print(np.nanstd(method_nlpd, axis=1))
