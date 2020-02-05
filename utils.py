import sys
import config
import numpy as np

windowTrain = [] #window based preprocessed train data
windowDev = [] #window based preprocessed dev data
windowTest = [] #window based preprocessed test data
Lengths = []

def read_data(fname, type, test):
    """
    open and preprocess data (train/ dev/test)
    :params: fname, type, test  type - 'pos'/'ner', test -'True'/'False', fname-path to file
    :return list - preprocessed file 
    """
    sentencesList = []    
    with open(fname) as input_file:
        sentence = []
        for line in input_file.read().split('\n'):
            if len(line) == 0:
                sentencesList.append( [("<s>", "S"), ("<s>", "S")] +
                        sentence +
                [("</s>", "S"), ("</s>", "S")])
                sentence = []                
                continue
            if type == 'ner':
               for words in line.strip().split():
                   if test:
                      text = words.strip().split() 
                      sentence.append((text[0],""))   
                   else:
                       text, label = words.strip().split("/",1)
                       sentence.append((text, label))
            if type == 'pos':
                if test:
                   text = line.strip().split() 
                   sentence.append((text[0],"")) 
                else:
                    text, label = line.strip().split(" ",1)
                    sentence.append((text, label)) 
        return sentencesList


testData = [item for sublist in read_data(sys.argv[7], sys.argv[4], True) for item in sublist] 

for i in range(len(testData)):
    if testData[i][1] != 'S':
       windowTest.append(([testData[i-2][0],testData[i-1][0],testData[i][0],testData[i+1][0],testData[i+2][0]]))

trainData = [item for sublist in read_data(sys.argv[5], sys.argv[4], False) for item in sublist]
devData = [item for sublist in read_data(sys.argv[6],sys.argv[4],False) for item in sublist]

for i in range(len(trainData)):
    if trainData[i][1] != 'S':
       windowTrain.append(([trainData[i-2][0],trainData[i-1][0],trainData[i][0],trainData[i+1][0],trainData[i+2][0]],trainData[i][1]))

for i in range(len(devData)): 
    if devData[i][1] != 'S':
       windowDev.append(([devData[i-2][0],devData[i-1][0],devData[i][0],devData[i+1][0],devData[i+2][0]],devData[i][1]))


if 'dev' in sys.argv[6] and sys.argv[4] == 'ner':
    with open(sys.argv[6]) as input_file:
         sentence = []
         for line in input_file.read().split('\n'):
             Lengths.append(len(line.strip().split(" "))) 


if 'dev' in sys.argv[6] and sys.argv[4] == 'pos':
    with open(sys.argv[6]) as input_file:
         sentence = []
         for line in input_file.read().split('\n'):
             if not len(line) == 0 or line == '' : 
                sentence.append(line)
             if len(line) == 0:
                Lengths.append(len(sentence))
                sentence = []

            
tag = [a[1] for a in trainData if not a[1] == '/O']
words = [a[0] for a in trainData]

chars = []
for word in words:
    for char in word:
        chars.append(char)

chars = set(chars)
chars.add("UUUNKKK")
config.VOCAB_CHARS = len(chars)

vocab = set(words)
vocab.add("UUUNKKK")
config.VOCAB = len(vocab)

tags = set(tag)
tags.add("UUUNKKK")


wordsPrefix = [a[0][:3] for a in trainData]
wordsSuffix = [a[0][-3:] for a in trainData]

vocabPrefix = set(wordsPrefix)
vocabPrefix.add("UUUNKKK")
config.VOCAB_PREFIX = len(vocabPrefix)

vocabSuffix = set(wordsSuffix)
vocabSuffix.add("UUUNKKK")
config.VOCAB_SUFFIX = len(vocabSuffix)

config.VOCAB = len(vocab)
config.OUTPUT_SIZE = len(tags)

charToId = { char: i for i, char  in enumerate(chars)} 
idToChar = { i: char for i, char in enumerate(chars)}

wordToId = { word: i for i, word  in enumerate(vocab)} 
idToWord = { i: word for i, word in enumerate(vocab)}

tagToId = { tag: i for i, tag in enumerate(tags)}
idToTaG = { i: tag for i, tag in enumerate(tags)}

idToSuffix = {i: suffix for i, suffix  in enumerate(vocabSuffix)} 
suffixToId = {suffix: i for i, suffix  in enumerate(vocabSuffix)}
 
idToPrefix = {i: prefix for i, prefix  in enumerate(vocabPrefix)} 
prefixToId = {prefix: i for i, prefix  in enumerate(vocabPrefix)} 


def getPrefixOfId(Id):
    return idToPrefix[Id]

def getIdOfPrefix(word):
    ans = None 
    if word in prefixToId:
       ans = prefixToId[word]
    else:
       ans = prefixToId["UUUNKKK"]   
    return ans 

def getSuffixOfId(Id):
    return idToSuffix[Id]

def getIdOfSuffix(word):
    ans = None 
    if word in suffixToId:
       ans = suffixToId[word]
    else:
       ans = suffixToId["UUUNKKK"]   
    return ans 

def getCharOfId(Id):
    return idToChar[Id]

def getIdOfChar(char):

    ans = None 
    if char in charToId:
       ans = charToId[char]
    else:
       ans = charToId['UUUNKKK']  
    return ans

def getWordOfId(Id):
    return idToWord[Id]

def getIdOfWord(word):
    ans = None 
    if word in wordToId:
       ans = wordToId[word]
    else:
       ans = wordToId["UUUNKKK"]   
    return ans 
  
def getTagOfId(Id):
    return idToTaG[Id]

def getIdOfTag(tag):  
    ans = None 
    if tag in tagToId:
       ans = tagToId[tag]
    else:
       ans = tagToId["UUUNKKK"]
    return ans 


wordsLengths = [[len(word) for word in words[0]] for words in windowTrain]
flattenedList = [y for x in wordsLengths for y in x]
config.WORD_MAX_LENGTH = np.max(flattenedList)















