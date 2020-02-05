import random
import string

pos_examples = []
neg_examples = []

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


def write_data(file_name, data):

    _file = open(file_name,'w')           
    _file.write('\n'.join(data)) 
    _file.close()


def generate_examples(N):
    """
    Concatinate sub strings
    :params:
    :return string: 
    """  
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
      pos_examples.append(pos_dc)
      neg_examples.append(neg_dc)
    write_data('pos_examples',pos_examples)
    write_data('neg_examples',neg_examples)


def generate_train_data():
    n_examples, p_examples = [], [] 
    for example in neg_examples:
        n_examples.append(example+' '+str(0))
    for example in pos_examples:
        p_examples.append(example+' '+str(1))

    train_data = n_examples+p_examples
    _file = open('train','w')           
    _file.write('\n'.join(train_data)) 
    _file.close()   

def generate_test_data():
    generate_examples(4)
    n_examples, p_examples = [], [] 
    for example in neg_examples:
        n_examples.append(example)
    for example in pos_examples:
        p_examples.append(example)

    test_data = n_examples+p_examples
    _file = open('test','w')           
    _file.write('\n'.join(test_data)) 
    _file.close()  

   
if __name__ == "__main__":
   generate_examples(4)
   generate_train_data()
   pos_examples = []
   neg_examples = []
   generate_test_data()





