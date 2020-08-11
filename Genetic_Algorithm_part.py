import random as rd
import numpy as np
from operator import itemgetter
import math


# dnn_list(n) custom_dnn class의 list
# stack(5*n) 0: num of DNN 1: neural num.list 2: hidden num.int 3: batch size 4:stack
# 상위 10개 하위 30% (추후 조정)

def descen_create(param_stack, dnn_list):
    stack = sorted(param_stack, key=itemgetter(4))
    batch_size_list = []
    hidden_list = []
    num_dnn = len(stack)
    # int(num_dnn/5-1)
    for i in range(10):
        batch_size_list.append(stack[-10:][i][3])
        hidden_list.append(stack[-10:][i][2])
    hidden_mean = np.mean(hidden_list)
    hidden_std = np.std(hidden_list)
    batch_size_mean = np.mean(hidden_list)
    batch_size_std = np.std(hidden_list)
    for t in range(int(num_dnn / 3 - 1)):
        if rd.random() < 0.01:  # mutant index = 0.1
            batch_size = rd.randint(1, 8)
        else:
            batch_size = int(batch_size_std * np.random.randn() + batch_size_mean)
        if rd.random() < 0.01:  # mutant index = 0.1
            hidden = rd.randint(2, 5)
        else:
            hidden = int(hidden_std * np.random.randn() + hidden_mean)
        neural_sample_1, neural_sample_2 = rd.sample(stack[-10:], 2)
        neural = []
        ln1, ln2 = [len(neural_sample_1), len(neural_sample_2)]
        s, m = [ln2, ln1] if ln1 > ln2 else [ln1, ln2]
        s_sample = neural_sample_2 if ln1 > ln2 else neural_sample_1
        for i in range(m):
            if i < s:
                if rd.random() < 0.01:  # mutant index = 0.1
                    neural.append(rd.randint(16, 32 * 32))
                else:
                    neural.append(
                        int(abs(neural_sample_1[i] - neural_sample_2[i]) / 4 * np.random.randn() \
                            + mean([neural_sample_1[i], neural_sample_2[i]])))
            else:
                if rd.random() < 0.01:  # mutant index = 0.1
                    neural.append(rd.randint(16, 32 * 32))
                else:
                    neural.append(s_sample[i])
        train_data, test_data, epoch = < 데이터
        불러오기 > (by batch_size)
        dnn_list[int(stack[t][0])] = Net(neural, hidden, train_data, test_data, epoch)
        update_stack(stack, int(stack[t][0]), neural, hidden)
    reset_stack(stack)


# stack create,reset,update
def create_stack(dnn_list):
    stack = [[i] for i in range(len(dnn_list))]
    for i in range(len(dnn_list)):
        stack[i].append(dnn_list[i].Get_neural)
        stack[i].append(dnn_list[i].Get_hidden)
        Stack[i].append(0)
    return stack


def reset_stack(stack):
    for i in stack:
        i[3] = 0


def update_stack(stack, position, neural, hidden):
    stack[position][1] = neural
    stack[position][2] = hidden
    stack[position][3] = 0


# -----------------------------------------------code start

def start(num):  # First Step
    rt = []
    for i in range(num):
        param2 = []  # neural_num
        param1 = rd.randint(2, 5)  # hidden_num
        param3 = rd.randint(1, 8)  # batch size
        for j in range(param2):
            param1.append(rd.randint(16, 32 * 32))
        train_data, test_data, epoch = < 데이터
        불러오기 > (by param3)
        rt.append(Net(param1, param2, train_data, test_data, epoch))
    return rt


# Second Step_(run - #stack) * n
# 50% - stack
# accuracy,speed - stack(r) return, if num = 3 then test three simultaneously
# reward 0:neural_num 1:time_reward 2:accuracy_reward 3: ...

def gene_stack(reward, r, num=3):
    for i in range(num):
        rewd = sorted(reward, key=itemgetter(i + 1))
        stack = np.zeros(len(reward))
        for i in rewd[0:int(len(reward) / 2)]:
            stack[int(i[0])] += r
    return stack


# num: stack 반복횟수

def Gene(num, list_DNN)
    stack_total = np.zeros(len(reward))
    num_dnn = len(list_DNN)
    param_list = [[i] for i in range(num_dnn)]
    for i in range(num):
        result_list = [[i] for i in range(num_dnn)]
        for j in range(len(list_DNN)):
            param = list_DNN[j].run()
            result_list[j].append(param[3])
            result_list[j].append(param[4])
            result_list[j].append(param[5])
            if i == 0:
                param_list[j].append(param[0])
                param_list[j].append(param[1])
                param_list[j].append(param[2])
        stack_total += gene_stack(result_list, 1, 3)
    param_stack = [i for i in param_list]
    for i in stack_total:
        param_stack.append(i)
    descen_create(param_stack, list_DNN)
