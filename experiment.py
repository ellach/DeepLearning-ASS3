import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import config_experiment as config
import random
from torch.autograd import Variable
import os
import sys

class LSTM(nn.Module):
    """
    """
    def __init__(self):
        """
        Initialize the model by setting up the layers.
        """
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(config.VOCAB, config.EMBED_DIM)
        self.lstm = nn.LSTM(config.EMBED_DIM, config.HIDDEN_DIM) 
        self.fc1 = nn.Linear( config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.fc2 = nn.Linear( config.HIDDEN_DIM, config.OUTPUT_SIZE)  
        self.tanh = nn.Tanh() 
        self.sigmoid = nn.Sigmoid() 


    def forward(self, inputs, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        embedding = self.embedding(inputs) 
        lstm_out, hidden = self.lstm(embedding.view(len(inputs), 1, -1), hidden)
        out = self.tanh(self.fc1(lstm_out[-1][-1])) 
        sig_out = self.sigmoid(self.fc2(out))
        return sig_out, hidden


    def init_hidden(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
 
        weight = next(self.parameters()).data
        hidden = (weight.new(1, 1, config.HIDDEN_DIM).zero_(),
                     weight.new(1, 1 , config.HIDDEN_DIM).zero_()) 
        return hidden


class MLP(object):
    def __init__(self):
        self.model = LSTM()
        self.loss_fn = nn.BCELoss()
        self.initialize()
        self.char_to_id={config.CHARS[i]:i for i in range(len(config.CHARS))}
        self.data_acc = []
        self.data_loss  = [] 

    def initialize(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.LR, weight_decay=1e-6, momentum=0.9, nesterov=True)
   
    def getAccuracy(self,y_hats, y): 
        good = 0
        if int(round(y_hats[0])) == int(y[0]):
           good += 1 
        return good 

    def epoch(self, dataset, optimize):

        for epoch in range(config.EPOCHS_EX):
            totalLoss,  goodAccuracy = 0, 0
            random.shuffle(dataset) 
            for data, label in dataset:
 
                self.model.zero_grad()  
                   
                hidden = self.model.init_hidden()   

                x_batch = [self.char_to_id[a] for a in data] 
                y_batch = [int(label)] 

                while len(x_batch) < config.INPUT_SIZE: 
                      x_batch.append(0)

                x = Variable(torch.LongTensor(x_batch))
                y = Variable(torch.LongTensor(y_batch)) 
                
                lstm_output, hidden = self.model(x, hidden)

                lstm_output = lstm_output.view(len(lstm_output))
                y = y.float() 
                loss = self.loss_fn(lstm_output, y)
                totalLoss += loss.data 

                loss.backward()
                self.optimizer.step()
                goodAccuracy += self.getAccuracy((lstm_output.data).numpy(), (y.data).numpy())

            self.data_acc.append(goodAccuracy)
            self.data_loss.append((totalLoss/len(dataset))) 
            print('   Epoch: ',epoch ,'  Loss : {0:.6f}'.format(totalLoss/len(dataset)), '  Accuracy : {0:.6f}'.format(goodAccuracy/len(dataset)))    
    
 
    def getGraph(self):

        plt.figure()
        plt.plot(range(config.EPOCHS_EX), self.data_acc)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.savefig('first_and_last_acc_graph.png') 

        plt.figure()
        plt.plot(range(config.EPOCHS_EX), self.data_loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig('first_and_last_loss_graph.png') 


    def test(self, dataset): 
        posAccuracy = 0 
        random.shuffle(dataset)  
        for data in dataset:

            hidden = self.model.init_hidden()
            x_batch = [self.char_to_id[a] for a in data] 
            x = Variable(torch.LongTensor(x_batch))    

            while len(x_batch) < config.INPUT_SIZE: 
                      x_batch.append(0)

            lstm_output, hidden = self.model(x, hidden)
            posAccuracy += int(round((lstm_output.data).numpy()[0])) 
            print('Accuracy:  {0:.6f}'.format(posAccuracy/len( (lstm_output.data).numpy())))  
 

if __name__ == "__main__":

   strings_lengths = [] 
   dataset = []
   dir_train = os.getcwd() +'/'+sys.argv[1] 
   with open(dir_train) as input_file:
        for line in input_file.read().split('\n'):
            if len(line) > 0:
               data, label = line.split(' ',1) 
               dataset.append((data, label))
               strings_lengths.append(len(data))
        config.INPUT_SIZE = max(strings_lengths)      
    
   model = MLP() 
   model.epoch(dataset, True)
   model.getGraph()

   if len(sys.argv) > 2:
      dir_test = os.getcwd() +'/'+sys.argv[2]
      strings_lengths = []
      test_dataset = []
      with open(dir_test) as input_file:
           for line in input_file.read().split('\n'):
               if len(line) > 0:
                  test_dataset.append(line)
                  strings_lengths.append(len(line))
           config.INPUT_SIZE = max(strings_lengths) 

      model.test(test_dataset)



