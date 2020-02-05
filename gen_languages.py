import random
import string
import os


def digits_generator(N):
    """
    Generate digits string in length N
    :params N: string length
    :return string: 
    """
    return  ''.join([str(random.randint(1, 9)) for i in range(N)])

def chars_generator(N):
    """
    Generate characters string in length N, with characters in range a-b
    :params N: string length
    :return string: 
    """
    assert N < 27
    return ''.join([string.ascii_letters[:N]]) 


def palindroms_generator(examples_num = 500):
    """
    gen palindromes good examples
    :param examples_num: num of examples
    :return: palindromes good examples
    """
    neg_examples = []
    pos_examples = []
    for i in range(examples_num):
        pos_cur_example = ''
        pe = '' 
        for _ in range(random.randint(1, 50)):
            pos_cur_example += str(random.randint(1, 2))
        pe = pos_cur_example + ''.join(reversed(pos_cur_example))
        pos_examples.append(pos_cur_example)

        neg_cur_example = ''
        for _ in range(random.randint(1, 50)):
            neg_cur_example += str(random.randint(1, 2)) 
        neg_examples.append(neg_cur_example)
 
    generate_train_data(pos_examples, neg_examples,'polindroms_train')

   
def first_and_last_generator(N):

    pos_examples, neg_examples = [], []
    pos_chars = chars_generator(int(N))
    t = list(pos_chars)   
    t[2], t[1] = t[1], t[2]
    neg_chars=''.join(t) 
    for _ in range(500):
      pos_dc, neg_dc = '', ''   
      for i in range(N): 
        pos_digits = digits_generator(random.randint(1,9))
        neg_digits = digits_generator(random.randint(1,9)) 

        pos_letters = ''.join([pos_chars[i] for _ in range(random.randint(1,9))])
        neg_letters = ''.join([neg_chars[i] for _ in range(random.randint(1,9))])
   
        pos_dc += pos_digits+pos_letters 
        pos_dc += pos_digits
        neg_dc += neg_digits+neg_letters  
        neg_dc += neg_digits
      
      pos_ex = list(pos_dc)    
      pos_ex[-1] = pos_ex[0]
      pe = ''.join(pos_ex)
      pos_examples.append(pe)
      neg_examples.append(neg_dc)
    generate_train_data(pos_examples, neg_examples,'first_and_last_train')



def repeate_first_char_generator(N):
    
    pos_examples, neg_examples = [], []
    pos_chars = chars_generator(int(N))
    t = list(pos_chars)   
    t[2], t[1] = t[1], t[2]
    neg_chars=''.join(t) 
    for _ in range(500):
      pos_dc, neg_dc = '', ''   
      for i in range(N): 
        pos_digits = digits_generator(random.randint(1,9))
        neg_digits = digits_generator(random.randint(1,9)) 
        pos_letters = ''.join([pos_chars[i] for _ in range(random.randint(1,9))])
        neg_letters = ''.join([neg_chars[i] for _ in range(random.randint(1,9))])   
        pos_dc += pos_digits+pos_letters 
        pos_dc += pos_digits
        neg_dc += neg_digits+neg_letters  
        neg_dc += neg_digits

      pos_ex = list(pos_dc)
      for i in range(len(pos_dc)):
          if i%3 == 0: 
             pos_ex[i] = pos_ex[0]

      pos_examples.append(''.join(pos_ex))
      neg_examples.append(neg_dc)
    generate_train_data(pos_examples, neg_examples,'repeate_first_char_train')


 
def generate_train_data(pos_examples, neg_examples,filename):
    p_examples, n_examples = [], []
    for example in neg_examples:
        n_examples.append(example+'   '+str(0))
    for example in pos_examples:
        p_examples.append(example+'   '+str(1))

    train_data = p_examples+n_examples
    _file = open(filename,'w')           
    _file.write('\n'.join(train_data)) 
    _file.close() 
  

def generate_test_data(pos_examples, neg_examples, filename):

    n_examples, p_examples = [], [] 
    for example in neg_examples:
        n_examples.append(example)
    for example in pos_examples:
        p_examples.append(example)

    test_data = p_examples+n_examples
    _file = open(filename,'w')           
    _file.write('\n'.join(test_data)) 
    _file.close()  

  
if __name__ == "__main__":
   '''
   if not os.path.exists('languages'):
      os.mkdir('languages')
   dir = os.getcwd()+'/languages'
   '''
   palindroms_generator(50)
   first_and_last_generator(4)
   repeate_first_char_generator(4)
   generate_test_data

   #generate_test_data()





