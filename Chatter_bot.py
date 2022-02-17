
# coding: utf-8

# In[1]:

from Queue import Queue
from threading import Thread
import subprocess
import re
import pandas as pd
import numpy as np
import json
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import np_utils

from anytree import Node, RenderTree
from anytree.dotexport import RenderTreeGraph
from anytree import Walker
from anytree import Resolver


# In[2]:

seq_length=10


# In[3]:

class conv_bot(object):
    
    def __init__(self, name):
        self.name=name
        self.forest=forest()
        self.utility_function="happy"
        self.emo_model=load_model("/home/kavin/Silo/College Work/AI/Donna Files/emotion_classifier_model.h5")
        emo_filename = "/home/kavin/Silo/College Work/AI/Donna Files/emotion_forward_weights-improvement-00-1.9928.hdf5"
        self.emo_model.load_weights(emo_filename)
        self.emo_model.compile(loss='categorical_crossentropy', optimizer='adam')
    
        self.pos_model=load_model("/home/kavin/Silo/College Work/AI/Donna Files/pos_tagger_model.h5")
        pos_filename = "/home/kavin/Silo/College Work/AI/Donna Files/tagger_weights-improvement-997-0.8726.hdf5"
        self.pos_model.load_weights(pos_filename)
        self.pos_model.compile(loss='categorical_crossentropy', optimizer='adam')

        self.gen_model=load_model("/home/kavin/Silo/College Work/AI/Donna Files/text_generator_model.h5")
        gen_filename = "/home/kavin/Silo/College Work/AI/Donna Files/text_512_lstm_gen_weights-improvement-969-0.0264.hdf5"
        self.gen_model.load_weights(gen_filename)
        self.gen_model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.greet()
        
    def greet(self):
        print("**************************")
        print("Hello. I am "+self.name+"!")
        print("**************************")
        print("\n")
        print("**PSST: I am a drunk lil. Its 5o'clock!! I might not make sense.!!")
        print("\n")
        print("\n")
                
        
    
    def session(self):
        self.greet()


# In[4]:

class conversationalist(object):
    
    def __init__(self, name):
        self.name=name
    
    def set_name(self, name):
        self._name=name


# In[5]:

class forest:
    def __init__(self):
        self.forest=[]
                
    def add_tree(self, tree):
        self.forest.append(tree)
        
    def find_tree_w_root(self, node):
        for tree in self.forest:
            if(tree.root == node):
                return tree
            
    def find_tree_w_node(self, node):
        walker=Walker()
        paths=[]
        for tree in self.forest:
            try:
                path=walker.walk(tree,node)[-1][-1]
                paths.append(path)
            except:
                pass
        if(len(paths)>0):
            return(True, paths)
        else:
            return(False, [])
        
    def find_tree_w_emotion(self, emo):
        trees=[]
        #node=Node()
        for tree in self.forest:
            try:
                for child in tree.root.descendants:
                    if(child.name==emo):
                        trees.append(tree)
                        node=child
            except:
                pass
        if(len(trees)>0):
            return(True, [trees,child])
        else:
            return(False, [])
        
    def print_all_trees(self):
        for tree in self.forest:
            for pre, fill, node in RenderTree(tree.root):
                print("%s%s" % (pre, node.name))
            


# In[6]:

# In[7]:

def create_process():
    p = subprocess.Popen("bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, universal_newlines=True)
    outQueue = Queue()
    outThread = Thread(target="", args=(p.stdout, outQueue))
    outThread.daemon = True
    outThread.start()
    return p


# In[8]:

def predef_questions():
    try:
        bot_questions=open("/home/kavin/Silo/College Work/AI/Donna Files/bot_basic_questions.txt",'r').readlines()
    except:
        return True
    for question in bot_questions:
        if(len(question.strip())>0):
            print(bot.name+": "+question)
            input_line = raw_input(participant.name+": ")
            if( "name" in question):
                try:
                    matched_name=re.match('^(\w+)$', input_line)
                    participant.name=matched_name.group(1)
                except:
                    matched_name=re.match('.+? (\w+)$', input_line)
                    participant.name=matched_name.group(1)
            if(input_line.lower() in ["exit", "quit", "stop", "bye"]):
                print("Goodbye, "+participant.name+"!")
                return False        
            proc.stdin.write(input_line)
    return True


# In[9]:

def getGeneratedText(input_line, bot):
    emotion=predictEmotion(input_line, bot)
    target_emotion=steerConversation(emotion, bot)
    #pos_pred=getPOSPrediction(input_line, bot)
    output_text=generateText(input_line, target_emotion)
    return output_text


# In[25]:

def predictEmotion(input_line, bot):
    tweets_x_cols=pd.read_csv("/home/kavin/Silo/College Work/AI/DataSets/colum_names.csv")
    input_line_vector=[0]*(len(tweets_x_cols)-1)
    for word in input_line:
        if(word in list(tweets_x_cols["0"])):
            input_line_vector[list(tweets_x_cols["0"]).index(word)]=1
    if(sum(input_line_vector)>0):
        prediction=bot.emo_model.predict(np.reshape(input_line_vector, (1,len(input_line_vector))))
    else:
        prediction=0
    
    with open("/home/kavin/Silo/College Work/AI/index_to_emotions.json") as data_file:    
        index_to_emotions = json.load(data_file)
    data_file.close()
    index_to_emotions[0]="Unknown"
    try:
        emot=index_to_emotions[str((list(prediction[0]).index(max(prediction[0]))))]
    except:
        emot="Unknown"        
    return emot


# In[11]:

def steerConversation(em, bot):
    target=bot.utility_function
    forest=bot.forest
    flag, tree=forest.find_tree_w_emotion(em)
    if(flag):
        paths=forest.find_tree_w_node(tree[1])
        shortest=min(paths, key=len)
        return shortest[1].name
    else:
        return "neutral"


# In[12]:

def getPOSPrediction(input_line, bot):
    #sentence="The weather is so nice today"
    pattern = re.split(' |.',input_line)
    indexed_input=[]
    with open("/home/kavin/Silo/College Work/AI/pos_index_to_words.json") as data_file:    
        index_to_words = json.load(data_file)
    data_file.close()
    with open("/home/kavin/Silo/College Work/AI/pos_bag_of_words.json") as data_file:    
        bag_to_words = json.load(data_file)
    data_file.close()
    try:
        ([indexed_input.append(bag_of_words[value.lower()]) for value in pattern])
    except:
        pass
        pattern=indexed_input
        pos_list=[]
        pattern+=list(np.zeros(int(math.ceil(len(pattern)/float(seq_length))*seq_length)-len(pattern)))
        for i in range(10):
            x = np.reshape(pattern, (len(pattern)/seq_length, seq_length, 1))
            x = x / float(303)
            prediction = bot.pos_model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = index_to_words[index]
            pos_list.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
    return pos_list
    


# In[ ]:

def generateText(input_line, target_emotion):
    pattern = re.split(' |.',input_line)
    mov_lines_file=open("/home/kavin/Silo/College Work/AI/DataSets/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus/movie_lines.txt",'r')
    mov_lines=mov_lines_file.readlines()
    dialogues=[]
    for mov_line in mov_lines:
        matched=re.match('.* \+\+\+\$\+\+\+ (.+)', mov_line)
        dialogues.append(matched.group(1))
    tokenized_dialogues=[]
    for line in dialogues:
        tokenized_dialogues.append(filter(None,re.split(' |!|:|;|\?|"|\-\-|\*|<.*?>|</.*?>', line.lower())))
    cleaned_dialogues=[]
    for line in tokenized_dialogues:
        line_list=[]
        for token in line:
            try:
                matched=re.match('([\*|\'|_|\-|\[]|)(.*)([\*|\'|_|\-|\]]|)', token)
                line_list.append(matched.group(2))
            except:
                pass
        cleaned_dialogues.append(line_list)
    all_dialogues=[]
    for line in cleaned_dialogues:
        all_dialogues+=line
    tokens=pd.Series(all_dialogues).unique()
    bag_of_words={}
    index_to_words={}
    index=1
    for word in tokens:
        bag_of_words[word]=index
        index_to_words[index]=word
        index+=1
    n_vocab = len(tokens)
    indexed_input=[]
    ([indexed_input.append(bag_of_words[value.lower()]) for value in pattern])
    pattern=indexed_input
    
    pattern+=list(np.zeros(int(math.ceil(len(pattern)/float(seq_length))*seq_length)-len(pattern)))
    created_text=""
    repeat_flag=True
    repeat_count=0
    for i in range(10):
        for i in range(10):
            x = np.reshape(pattern, (len(pattern)/seq_length, seq_length, 1))
            x = x / float(n_vocab)
            prediction = bot.gen_model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = index_to_words[index]
            created_text+=result+" "
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        if(predictEmotion(created_text, bot)):
            break
        
    return created_text


# In[14]:

proc=create_process()
bot=conv_bot("Donna")


# In[ ]:




# In[21]:

participant=conversationalist("You")
continue_flag=predef_questions()
while(continue_flag):
    input_line = raw_input(participant.name+": ")
    if(input_line.lower() in ["exit", "quit", "stop","bye"]):
        print("Goodbye, "+participant.name+"!")
        continue_flag=False
    else:
        output = getGeneratedText(input_line, bot)
        print(bot.name+": "+output)
    proc.stdin.write(input_line)
    


# In[ ]:



