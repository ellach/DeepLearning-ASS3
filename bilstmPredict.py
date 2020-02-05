from bilstmTrain import biLSTM 
from torch.autograd import Variable
import torch
import sys
import utils
import config
import numpy as np

if __name__ == "__main__":

   model_type = sys.argv[1]
   model_file = sys.argv[2]
   test_data = sys.argv[3]
   out_file = sys.argv[4]

   model = biLSTM()
   model.load_state_dict(torch.load(model_file))
   pred_data = [] 
   dataset = utils.windowTest
   for i in range(0, len(dataset)-config.BATCH_SIZE, config.BATCH_SIZE):
   
       if model_type in  ['a','b']: #words, characters 
          batch = dataset[i:i+config.BATCH_SIZE]              
          x_batch = [[utils.getIdOfWord(w) for w in a] for a in batch] 
          x = Variable(torch.LongTensor(x_batch))
          predictions, _ = model(x , model_type) 
          #predictions = prediction.view(100, -1)
           
       if model_type == 'c': #subwords 

          batch = dataset[i:i+config.BATCH_SIZE]       
          x_batch_prefix = [[utils.getIdOfPrefix(w[3:]) for w in a] for a in batch]  
          x_batch_suffix = [[utils.getIdOfSuffix(w[:-3]) for w in a] for a in batch]
          x_prefix = Variable(torch.LongTensor(x_batch_prefix))
          x_suffix = Variable(torch.LongTensor(x_batch_suffix))
          predictions, _ = model((x_prefix, x_suffix) , model_type) 

       if model_type == 'd': #conc

          batch = dataset[i:i+config.BATCH_SIZE]              
          x_batch = [[utils.getIdOfWord(w) for w in a] for a in batch]    
          x = Variable(torch.LongTensor(x_batch))
          prediction, _ = model(x , model_type) 
          predictions = prediction.view(100, -1)

       pred_tags = []
       for i in range(config.BATCH_SIZE):
           holder = []
           exps = [np.exp(item) for item in ((torch.flatten(predictions[i]).data).numpy()).tolist()]
           exp_sum = np.sum(exps)  
           for  pred in predictions[i]:
                res = exp_sum + np.exp((pred.data).numpy())
                Id = np.max(res)  
                if int(round(Id)) < len(utils.tags):       
                   holder.append( utils.getTagOfId(int(round(Id)))) 
           pred_tags.append(holder) 
       pred_data.append((batch, pred_tags))

   out = []
   for sentences, tags in pred_data:
       for i in range(config.BATCH_SIZE):
         for word,tag in zip(sentences[i], tags[i]):
             out.append(word + " " + tag+ "\n")
         out.append("\n")

   file_name = 'test4.'+out_file
   f = open(file_name,"w")
   f.write("".join(out))
   f.close()


