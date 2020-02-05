import torch
import torch.nn as nn
import config
import random
from torch.autograd import Variable
import numpy as np
import utils
import sys


class biLSTM(nn.Module):
    """
    """
    def __init__(self):
        """
        Initialize the model by setting up the layers.
        """
        super(biLSTM, self).__init__()
        
        # Words
        self.embedding = nn.Embedding(config.VOCAB, config.EMBED_DIM) 
        self.embedding.shape = torch.Tensor(config.BATCH_SIZE, 5*config.EMBED_DIM)
        self.lstm_words = nn.LSTM(5*config.EMBED_DIM, config.HIDDEN_DIM, num_layers=1,bidirectional=True)
       
        # Letters
        self.embedding_chars = nn.Embedding(config.VOCAB_CHARS, config.EMBED_DIM)
        self.embedding_chars.shape = torch.Tensor(config.BATCH_SIZE, config.WORD_MAX_LENGTH*config.EMBED_DIM) 
        self.lstm_chars = nn.LSTM(config.WORD_MAX_LENGTH*config.EMBED_DIM, config.HIDDEN_DIM, num_layers=1, bidirectional=True)

        #Subwords  
        self.embeds_prefix = nn.Embedding(config.VOCAB, config.EMBED_DIM) 
        self.embeds_prefix.shape = torch.Tensor(config.BATCH_SIZE, 5*config.EMBED_DIM)
        self.lstm_prefix = nn.LSTM(5*config.EMBED_DIM, config.HIDDEN_DIM, num_layers=1, bidirectional=True) 

        self.embeds_suffix = nn.Embedding(config.VOCAB, config.EMBED_DIM)
        self.embeds_suffix.shape = torch.Tensor(config.BATCH_SIZE, 5*config.EMBED_DIM) 
        self.lstm_suffix = nn.LSTM(5*config.EMBED_DIM, config.HIDDEN_DIM, num_layers=1, bidirectional=True)
          
        # Concatination
        self.embedding_concat_words = nn.Embedding(config.VOCAB, config.EMBED_DIM)
        self.embedding_concat_words.shape = torch.Tensor(config.BATCH_SIZE, 5*config.EMBED_DIM)

        self.embedding_concat_chars  = nn.Embedding(config.VOCAB, config.EMBED_DIM)
        self.embedding_concat_chars.shape = torch.Tensor(config.BATCH_SIZE, 5*config.WORD_MAX_LENGTH*config.EMBED_DIM) 
  
        self.lstm_concat_dim = (5*config.EMBED_DIM + 5*config.WORD_MAX_LENGTH*config.EMBED_DIM)
        self.lstm_concat = nn.LSTM(self.lstm_concat_dim, config.HIDDEN_DIM, num_layers=1, bidirectional=True)
         
        self.fc = nn.Linear(2*config.HIDDEN_DIM, config.OUTPUT_SIZE) 
        self.softmax = nn.LogSoftmax(dim=0) 


    def init_hidden(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and loss_fnl state of LSTM
        
        weight = next(self.parameters()).data
        hidden = (weight.new(config.N_LAYERS, 1, config.HIDDEN_DIM).zero_(),
                     weight.new(config.N_LAYERS, 1 , config.HIDDEN_DIM).zero_()) 
        return hidden
    

    def forward(self, inputs, mode):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        self.hidden = self.init_hidden()   

        if mode == 'a':

           embedding_word = self.embedding(inputs).view(self.embedding.shape.size())
           lstm_out, self.hidden = self.lstm_words(embedding_word.view(len(inputs), 1, -1), self.hidden)
           softmax_out = self.softmax(self.fc(lstm_out)) 

        if mode == 'b':
           
           embed_chars = self.embedding_chars(inputs).view(self.embedding_chars.shape.size()) 
           lstm_out_chars, self.hidden = self.lstm_chars(embed_chars.view(len(inputs), 1, -1),self.hidden)  
           softmax_out = self.softmax(self.fc(lstm_out_chars))

        if mode == 'c': 

           embedding_prefix = self.embeds_prefix(inputs[0]).view(self.embeds_prefix.shape.size())
           lstm_out_prefix, self.hidden = self.lstm_prefix(embedding_prefix.view(len(inputs[0]), 1, -1), self.hidden)
           embedding_suffix = self.embeds_suffix(inputs[1]).view(self.embeds_suffix.shape.size())
           lstm_out_suffix, self.hidden = self.lstm_suffix(embedding_suffix.view(len(inputs[1]), 1, -1), self.hidden)
           lstm_out = lstm_out_prefix+lstm_out_suffix
           softmax_out = self.softmax(self.fc(lstm_out)) 

        if mode == 'd': 

           embedding_c_words = self.embedding_concat_words(inputs[0]).view(self.embedding_concat_words.shape.size())
           embedding_c_chars = self.embedding_concat_chars(inputs[1]).view(self.embedding_concat_chars.shape.size())
           concat_input = torch.cat((embedding_c_words, embedding_c_chars),1)
           lstm_out, self.hidden =  self.lstm_concat(concat_input.view(100 , 1, -1), self.hidden)
           softmax_out = self.softmax(self.fc(lstm_out)) 

        return  softmax_out, self.hidden


class Solver(object):
    def __init__(self):
      """
      solver 
      """ 
      self.model = biLSTM()
      self.loss_fn = nn.CrossEntropyLoss()  
      self.initialize()
      self.acc_data_plots = [] 

    def initialize(self):
        """
        initialize optimizer for backprop 
        """  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LR) 

    def epoch(self, dataset, optimize, mode):
        """
        run model N iterations
        :params dataset, optimizer: 
        """ 
        accuracyDataSet = []
        lossDataSet = []
        for epoch in range(config.EPOCHS):
            totalLoss, goodAccuracy, totalAccuracy = 1, 1, 1
            random.shuffle(dataset) 
            for i in range(0, len(dataset)-config.BATCH_SIZE, config.BATCH_SIZE):
           
                if optimize:
                   self.model.zero_grad() 
                
                if mode == 'c':
           
                   batch = dataset[i:i+config.BATCH_SIZE]
                    
                   x_batch_prefix = [[utils.getIdOfPrefix(w[3:]) for w in a] for a,b in batch]  
                   x_batch_suffix = [[utils.getIdOfSuffix(w[:-3]) for w in a] for a,b in batch]
                   y_batch = [utils.getIdOfTag(b) for a,b in batch] 
                   x_prefix = Variable(torch.LongTensor(x_batch_prefix))
                   x_suffix = Variable(torch.LongTensor(x_batch_suffix))
                   y = Variable(torch.LongTensor(y_batch))

                   lstm_output, _ = self.model((x_prefix, x_suffix) ,  mode) 
     
                if mode in ['a','b']: 

                   batch = dataset[i:i+config.BATCH_SIZE]              
                   x_batch = [a for a,b in batch]
                   y_batch = [b for a,b in batch]
                   x = Variable(torch.LongTensor(x_batch))
                   y = Variable(torch.LongTensor(y_batch))
                   lstm_output, _ = self.model(x,  mode) 
            
                if mode == 'd':

                   batch = dataset[i:i+config.BATCH_SIZE]
                   x_batch_words = [[utils.getIdOfWord(w) for w in a] for a,b in batch]
                   x_batch_chars = [[[utils.getIdOfChar(c) for c in word] for word in words[0]] for words in batch]
                   x_batch_char = [item for sublist in x_batch_chars for item in sublist]
                   lengths = [len(item) for item in x_batch_char]
                   x_batch_c = []
                   length = np.max(lengths) 

                   for i in range(len(x_batch_char)):
                       if len(x_batch_char[i]) > config.WORD_MAX_LENGTH:
                          x_batch_c.append(x_batch_char[i][:config.WORD_MAX_LENGTH])

                       else:
                          while len(x_batch_char[i]) < config.WORD_MAX_LENGTH:
                                x_batch_char[i].append(0)                
                          x_batch_c.append(x_batch_char[i]) 
                   
                   x_words = Variable(torch.LongTensor(x_batch_words))
                   x_chars = Variable(torch.LongTensor(x_batch_c))
                   y_batch = [utils.getIdOfTag(b) for a,b in batch]
                   y = Variable(torch.LongTensor(y_batch))
                   lstm_output, _ = self.model((x_words,x_chars),  mode)                   
 
                lstm_output = lstm_output.view(100, -1) 
                loss = self.loss_fn(lstm_output, y)
                totalLoss += (loss.data).numpy()

                if optimize:
                   self.optimizer.zero_grad() 
                   loss.backward()
                   self.optimizer.step()

                goodAccuracy, totalAccuracy = self.getAccuracy((lstm_output.data).numpy(), (y.data).numpy(), mode)
                goodAccuracy += goodAccuracy
                totalAccuracy += totalAccuracy

            accuracyDataSet.append(goodAccuracy/totalAccuracy) 
            lossDataSet.append( totalLoss/(len(dataset)/config.BATCH_SIZE) )
            print('Model:  ',mode,'   Epoch: ',epoch ,'  Loss : {0:.6f}'.format(totalLoss/(len(dataset)/config.BATCH_SIZE)), '  Accuracy : {0:.6f}'.format( goodAccuracy/totalAccuracy ) )   

        if not optimize:
           self.acc_data_plots.append((accuracyDataSet, mode))


    def write(self): 
        f = open(self.acc_data_plots[0][1]+'.txt', "w")
        for data in self.acc_data_plots[0][0]:
            f.write(str(data)+'\n')
        for length in  utils.Lengths:
            f.write(str(length)+'\n')
        f.close() 


    def getAccuracy(self,y_hats, y, mode):

        good, bad = 0, 0
        if not mode == 'b':
           for i in range(0, config.BATCH_SIZE):
             y_hat = np.argmax(y_hats[i])
             y_hat_ = utils.getTagOfId(y_hat)
             y_ = utils.getTagOfId(y[i]) 
             if y_hat_ == y_:
                good += 1
             else: 
                bad += 1

        if mode == 'b':
          for i in range(0, config.BATCH_SIZE):
            y_hat = np.argmax(y_hats[i])
            y_hat_ = utils.getTagOfId(y_hat)
            y_ = utils.getTagOfId(y[i]) 
            if y_hat_ == y_:
               good += 1
            else: 
                bad += 1 
 
        return good, (good+bad)


def save(modelFile, model):
    torch.save(model.state_dict(),modelFile)


if __name__ == "__main__":
 
 solver = Solver()
 if sys.argv[1] == 'a':

    print('Start training ...')
    # words dataset train
    wordsDataset = []
    for a,b in utils.windowTrain:
        wordsDataset.append(([utils.getIdOfWord(w) for w in a], utils.getIdOfTag(b)))
    solver.epoch(wordsDataset, True, 'a')
    save(sys.argv[3], solver.model)

    print('Start validation ...')
    # words dataset dev
    wordsDatasetDev = []
    for a,b in utils.windowDev:
        wordsDatasetDev.append(([utils.getIdOfWord(w) for w in a], utils.getIdOfTag(b)))
    solver.epoch(wordsDatasetDev, False, 'a')
    solver.write()

 if sys.argv[1] == 'c':

    print('Start training ...')
    # subwords dataset train
    solver.epoch(utils.windowTrain, True, 'c') 

    save(sys.argv[3], solver.model)

    print('Start validation ...')
    # subwords dataset dev
    solver.epoch(utils.windowDev, False, 'c') 
    solver.write()

 if sys.argv[1] == 'b':

    print('Start training ...')
    # chars dataset train
    wordsLengths = [[len(word) for word in words[0]] for words in utils.windowTrain]
    flattenedList = [y for x in wordsLengths for y in x]
    config.WORD_MAX_LENGTH = np.max(flattenedList)

    lettersDataset = []
    for words in utils.windowTrain:
        sublist = []
        for word in words[0]:
            while len(word) < config.WORD_MAX_LENGTH:
                  word += 'a'  
            sublist = [utils.getIdOfChar(c) for c in word]
            lettersDataset.append((sublist, utils.getIdOfTag(words[1])))  
    solver.epoch(lettersDataset, True,'b') 
    save(sys.argv[3], solver.model)

    print('Start validation ...')
    # chars dataset dev

    lettersDatasetDev = []
    for words in utils.windowDev:
        sublist = []
        for word in words[0]:
            if len(word) > config.WORD_MAX_LENGTH:
               subword  = list(word) 
               sublist =  [utils.getIdOfChar(c) for c in subword[config.WORD_MAX_LENGTH:]] 
               word = ''.join(subword[: -(len(word) - config.WORD_MAX_LENGTH)])
               while len(word) < config.WORD_MAX_LENGTH:
                     word += 'a'   
               sublist = [utils.getIdOfChar(c) for c in word]
               lettersDatasetDev.append((sublist, utils.getIdOfTag(words[1])))
    solver.epoch(lettersDatasetDev, False,'b') 
    solver.write()

if sys.argv[1] == 'd':

   solver.epoch( utils.windowTrain, True,'d')
   save(sys.argv[3], solver.model)

   solver.epoch(utils.windowDev, False,'d')
   solver.write() 




