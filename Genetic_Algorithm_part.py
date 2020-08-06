# import
import random as rd
import numpy as np
from operator import itemgetter
import math


def mean(values):  # math function
    if len(values) == 0:
        return None
    return sum(values, 0.0) / len(values)


def standardDeviation(values, option=0):
    if len(values) < 2:
        return None
    sum = 0.0
    meanValue = mean(values)
    for i in range(0, len(values)):
        diff = values[i] - meanValue
        sum += diff * diff
    sd = math.sqrt(sum / (len(values) - option))
    return sd


class custom_dnn:  # dnn class_Makeshift
    def __init__(self, n, h):
        self.neural = n
        self.hidden = h
        self.accuracy = 0

    def Get_neural(self):
        return self.neural

    def Get_hidden(self):
        return self.hidden

    def Get_accuracy(self):
        for i in range(10): # step
            print("End")
        return self.accuracy
        
  """
  dnn_list(n) custom_dnn class의 list
  stack(4*n) 0: num of DNN 1: neural num.list 2: hidden num.int 3: stack
  상위 10개 하위 30% (추후 조정)
  """
def descen_create(stack,dnn_list):
    stck = sorted(stack, key=itemgetter(3))
    descen_hidden_list = []
    num_dnn = len(stack)
    #int(num_dnn/5-1)
    for i in range(10):
        descen_hidden_list.append(stck[-10:][i][2])
    descen_hidden_mean = mean(descen_hidden_list)
    descen_hidden_standardDeviation = standardDeviation(descen_hidden_list,0)
    for t in range(int(num_dnn/3-1)):
        descen_hidden = int(descen_hidden_standardDeviation * np.random.randn() + descen_hidden_mean)
        descen_neural_sample_1, descen_neural_sample_2 = rd.sample(stck[-10:],2)
        descen_neural = []
        ln1,ln2 = [len(descen_neural_sample_1),len(descen_neural_sample_2)]
        s,m = [ln2,ln1] if ln1>ln2 else [ln1,ln2]
        s_sample= descen_neural_sample_2 if ln1>ln2 else descen_neural_sample_1
        for i in range(m):
            if i<s:
                descen_neural.append(int(abs(descen_neural_sample_1[i]-descen_neural_sample_2[i])/4 * np.random.randn() \
                                         + mean([descen_neural_sample_1[i],descen_neural_sample_2[i]])))
            else:
                descen_neural.append(s_sample[i])
        dnn_list[int(stck[t][0])] = custom_dnn(descen_neural,descen_hidden)
        update_stack(stack,int(stck[t][0]),descen_neural,descen_hidden)
    reset_stack(stack)
    

def create_stack(dnn_list):  # stack create,reset,update
    stack = [ [i] for i in range(len(dnn_list))]
    for i in range(len(dnn_list)):
        stack[i].append(dnn_list[i].Get_neural)
        stack[i].append(dnn_list[i].Get_hidden)
        stack[i].append(0)
    return stack

def reset_stack(stack):
    for i in stack:
        i[3] = 0
        
def update_stack(stack,position,neural,hidden):
    stack[position][1] = neural
    stack[position][2] = hidden
    stack[position][3] = 0

  """
  50% - stack
  accuracy,speed - stack(r) return, if num = 2 then test two simultaneously
  reward 0:neural_num 1:time_reward 2:accuracy_reward 3: ...
  """
def gene_stack(reward,r,num = 1):
    for i in range(num):
        rewd = sorted(reward, key=itemgetter(i+1))
        stack = np.zeros(len(reward))
        for i in rewd[0:int(num_dnn/2)]:
            stack[int(i[0])] += r
    return stack
