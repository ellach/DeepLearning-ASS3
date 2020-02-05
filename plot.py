import matplotlib.pyplot as plt
import os
import sys
import numpy as np 

data_plots = []
lengths = []


def number_of_sentences(lengths):
    arr = []
    count = 0 
    for i in range(0,len(lengths)):
        if not i%500 == 0:
           count += 1 
        else:    
           arr.append(int(count/100))
           count += 1
    return arr[:6]  


for i in range(1,len(sys.argv)):
    lengths_holder = [] 
    data = []
    data_avg = []  
    j = 0  
    with open(sys.argv[i],'r') as f:
         for line in f.read().split('\n'):
             if len(line) > 0: 
                if j < 20:
                   acc = line.strip().split(' ') 
                   data.append(float(acc[0]))
                if j > 20:
                   length_of_sentence = line.strip().split('\n')
                   lengths_holder.append(int(length_of_sentence[0]))
                j += 1
         model_name = os.path.splitext(os.path.basename(sys.argv[i]))[0]
         lengths = [ln for ln in number_of_sentences(lengths_holder)]
         data_avg = [np.average(data[i:i+3]) for i in range(0,len(data) -3 ,3)]
         data_plots.append((data_avg, model_name))


if len(data_plots) > 0:

   a = data_plots[0][0]
   b = data_plots[1][0]
   c = data_plots[2][0]   
   d = data_plots[3][0] 

   plt.figure()
   plt.title("dev accuracy per number of lines trained")
   plt.xlabel("number of sentences/100 ")
   plt.ylabel("accuracy")
   
   plt.plot(lengths, a, label = str(data_plots[0][1])) 
   plt.plot(lengths, b, label = str(data_plots[1][1])) 
   plt.plot(lengths, c,label = str(data_plots[2][1])) 
   plt.plot(lengths, d,label = str(data_plots[3][1]))  

   plt.ylim(0,1)
   plt.xlim(0,30,5) 
   plt.legend()
   plt.savefig('plot.png')







