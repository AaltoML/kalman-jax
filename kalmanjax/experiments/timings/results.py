import pickle
import numpy as np

task_list = ['heteroscedastic', 'coal', 'banana', 'binary', 'audio', 'aircraft', 'rainforest']

method_timings = np.zeros([10, 6])
for method in range(10):
    for task_num in range(6):
        task = task_list[task_num]
        if (task_num == 4) and method in [4, 5, 7, 9]:
            method_timings[method, task_num] = np.nan
        else:
            with open("output/" + str(task) + "_" + str(method) + ".txt", "rb") as fp:
                result = pickle.load(fp)
                # print(result)
                method_timings[method, task_num] = result

# for fold in range(10):
#     with open("output/" + str(15) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#         print(pickle.load(fp))

np.set_printoptions(precision=3)
print(method_timings[:, :-1])
# print(np.nanmean(method_nlpd, axis=1))
# np.set_printoptions(precision=2)
# print(np.std(method_nlpd, axis=1))
# print(np.nanstd(method_nlpd, axis=1))
