import random as rd
import numpy as np
from operator import itemgetter
import dnn
from grayscale_colorization import binary_util as butil


class genetic:
    def __init__(self, model_n):
        self.model_n = model_n
        self.model_list = [[0]] * model_n

    def start(self):  # First Step
        rt = []
        for i in range(self.model_n):
            param1 = rd.randint(2, 5)  # hidden_layer_num
            param2 = []  # neural_num
            param3 = rd.randint(1, 8)  # batch size

            for j in range(param1):
                param2.append(rd.randint(16, 32 * 32))

            train_data = butil.load("trainset", param3)
            test_data = butil.load("testset", param3)
            epoch = 5
            rt.append([dnn.Net(param1, param2, train_data, test_data, epoch), [param1, param2, param3]])
        return rt

    def create(self, param_list):
        rt = []
        for i in range(self.model_n):
            ns1, ns2 = rd.sample(param_list, 2)
            ln1 = ns1[0]
            ln2 = ns2[0]
            b, s, b_sample = [ln1, ln2, ns1] if ln1 >= ln2 else [ln2, ln1, ns2]
            neural = []
            if rd.random() < 0.05:  # mutant index = 0.05
                batch_size = rd.randint(1, 8)
            else:
                rate = rd.random()
                if rate > 0.66:
                    batch_size = ns1[2]
                elif rate > 0.33:
                    batch_size = ns2[2]
                else:
                    batch_size = int((ns1[2] + ns2[2]) / 2)

            if rd.random() < 0.05:  # mutant index = 0.05
                hidden = rd.randint(2, 5)
            else:
                rate = rd.random()
                if rate > 0.66:
                    hidden = ns1[0]
                elif rate > 0.33:
                    hidden = ns2[0]
                else:
                    hidden = int((ns1[0] + ns2[0]) / 2)

            for j in range(hidden):
                if j < s:
                    if rd.random() < 0.05:  # mutant index = 0.05
                        neural.append(rd.randint(16, 32 * 32))
                    else:
                        neural.append(
                            int(np.std([ns1[1][j], ns2[1][j]]) * np.random.randn() + np.mean([ns1[i], ns2[i]])))

                else:
                    if rd.random() < 0.01:  # mutant index = 0.1
                        neural.append(rd.randint(16, 32 * 32))
                    else:
                        neural.append(b_sample[i])

            train_data = butil.load("trainset", batch_size)
            test_data = butil.load("testset", batch_size)
            epoch = 5
            rt.append([dnn.Net(hidden, neural, train_data, test_data, epoch), [hidden, neural, batch_size]])
        return rt

    def stack(self, result_list, num=3):
        stack = [0] * self.model_n

        for i in range(num):
            rewd = sorted(result_list, key=itemgetter(i + 1))

            for j in rewd[0:(self.model_n / 2)]:
                if i == 0:
                    stack[int(j[0])] += 1  # accuracy
                elif i == 1:
                    stack[int(j[0])] += 0.5  # train time
                else:
                    stack[int(j[0])] += 0.1  # test time

        return stack

    def competition(self, model_list):
        active_table = [1] * self.model_n
        stack_table = [0] * self.model_n

        while True:
            result_list = [[i] for i in range(self.model_n)]

            for i in range(self.model_n):
                if active_table[i]:
                    param = model_list[i][0].run()
                    result_list[i].append(param[3])
                    result_list[i].append(param[4])
                    result_list[i].append(param[5])

            stack_table += genetic.stack(result_list, 3)

            active_n = 0
            for j in range(self.model_n):
                if stack_table[j] >= 5:
                    active_table[j] = 0
                else:
                    active_n += 1

            if active_n <= (self.model_n * 0.3):
                break

        param_list = [i[1] for i in model_list]
        return param_list
