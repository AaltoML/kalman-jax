import pickle
import numpy as np

method_nlpd = np.zeros([17, 10])
for method in range(17):
    for fold in range(10):
        with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
            result = pickle.load(fp)
            print(result)
            method_nlpd[method, fold] = result

# for fold in range(10):
#     with open("output/" + str(15) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#         print(pickle.load(fp))

np.set_printoptions(precision=3)
print(np.mean(method_nlpd, axis=1))
# print(np.nanmean(method_nlpd, axis=1))
np.set_printoptions(precision=2)
print(np.std(method_nlpd, axis=1))
# print(np.nanstd(method_nlpd, axis=1))
