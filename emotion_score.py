
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re
import csv


# In[6]:

response_emotion_df=None


# In[2]:

response_emotion_df=pd.read_csv("DataSets/text_emotion.csv")


# In[11]:

response_emotion_df.head()


# In[3]:

unique_emotions=pd.Series(response_emotion_df["sentiment"]).unique()


# In[4]:

bag_of_emotions={}
index_to_emotions={}


# In[5]:

index=1
for emotion in unique_emotions:
    bag_of_emotions[emotion]=index
    index_to_emotions[index]=emotion
    index+=1
#n_chars = len(tokenized)


# In[6]:

all_response_words=[]


# In[7]:

for response in response_emotion_df["content"]:
    all_response_words.append(list(filter(None,re.split(' |\'|!|:|;|\?|"|\(|\)|/|\.+?|&|\-\-|\*|<.*?>|</.*?>', response.lower()))))


# In[ ]:

print(all_response_words[0])


# In[8]:

cleaned_responses=[]
all_words=[]


# In[9]:

for response in all_response_words:
    response_words=[x for x in response if not x.startswith(('@','#')) and not x.isdigit()]
    cleaned_responses.append(response_words)
    all_words+=response_words


# In[ ]:

print(cleaned_responses[0])


# In[10]:

all_unique_words=pd.Series(all_words).unique()


# In[11]:

bag_of_words={}
index_to_words={}
index=1
for word in all_unique_words:
    bag_of_words[word]=index
    index_to_words[index]=word
    index+=1
#n_chars = len(tokenized)
n_vocab = len(all_unique_words)
print(n_vocab)


# In[ ]:

bag_of_words


# In[32]:

def setNeuralNet(input_neurons, hidden_neurons, output_neurons):
    neural_net={}
    neural_net["hidden"]=np.random.randn(hidden_neurons,input_neurons)
    neural_net["output"]=np.random.randn(output_neurons,hidden_neurons)
    return neural_net


# In[33]:

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward_propagation(net, inputs):
    hidden=np.dot(net["hidden"], inputs)
    hidden[hidden<0] = 0
    out=np.dot(net["output"], sigmoid(hidden))
    return sigmoid(hidden), sigmoid(out)

def diff(output):
    return output*(1-output)

def back_propagate(neuron_outputs,y):
    out_errors=[]
    hidden_errors=[]
    hidden_deltas=[0]*len(nnet["hidden"])
    for y_hat in neuron_outputs:
        out_err=(y-y_hat)*diff(y_hat)
        out_errors.append(out_err)
        for out_weight in nnet["output"]:
            weights=[x * out_err for x in out_weight]
            hidden_errors.append([x *diff(y_hat) for x in weights])
    for delta in hidden_errors:
        hidden_deltas= [sum(x) for x in zip(hidden_deltas, delta)]
    return hidden_deltas, out_errors

def update_weights(layer, deltas, learning_rate, inputs):
    neuron_index=0
    new_weights=[]
    for neurons in nnet[layer]:
        new_neuron_weights=[]
        index=0
        for weights in neurons:
            new_neuron_weights.append(weights+(learning_rate*deltas[neuron_index]*inputs[index]))
            index+=1
        neuron_index+=1
        new_weights.append(new_neuron_weights)
    nnet[layer]=new_weights

def getAccuracy(y, y_hat):
    correct=0
    for index in range(1,len(y)):
        if y[index]==y_hat[index]:
            correct+=1
    print("Number of instances tested: "+ str(len(y)))
    print("Number of instances classified correctly: "+ str(correct))
    return(round(float(correct)/len(y)*100,2))


# In[41]:

def main(nnet, df, epochs):
    learning_rate=0.5
    #epochs=100
    #df=pd.read_csv("E:/College Work/CS534 - AI/Chandrasekaran_Kavin_Assign4/heart_data.csv")
    is_train=np.random.rand(len(df))<0.8
    train, test = df[is_train], df[~is_train]
    #del train["is_train"]
    #del test["is_train"]
    for epoch in range(epochs):
        rse=0
        for train_instance in train.itertuples():
            input_in=train_instance[1:-1]
            hidden_out, output_out=forward_propagation(nnet,input_in)
            y=train_instance[-1]
            rse+=sum((y-output_out)**2)
            hidden_deltas, out_deltas=back_propagate(output_out, y)
            update_weights("hidden",hidden_deltas,learning_rate, input_in)
            update_weights("output",out_deltas,learning_rate, hidden_out)
        #print('Epoch= %d, RSE= %.2f' % (epoch, rse))
    y_hat=[]
    y=[]
    for test_instance in test.itertuples():
        input_in=test_instance[1:-1]
        hidden_out, output_out=forward_propagation(nnet,input_in)
        y.append(test_instance[-1])
        if(output_out>0.5):
            y_hat.append(1)
        else:
            y_hat.append(0)
    accuracy=getAccuracy(y, y_hat)
    print("Accuracy for ANN with backpropagation is %: "+str(accuracy))


# In[35]:

nnet=setNeuralNet(len(all_unique_words), 10000, 8)


# In[ ]:

main(nnet,responses_df[:1000],1000)


# In[ ]:

responses_x=pd.DataFrame(columns=list(all_unique_words)+["emotion_y"])


# In[13]:

responses_x_cols=list(all_unique_words)+["emotion_y"]


# In[16]:

pd.DataFrame(responses_x_cols).to_csv("DataSets/colum_names.csv")


# In[ ]:

store['responses_emotion']=responses_x


# In[ ]:

store


# In[ ]:

table = store.root.responses_emotion


# In[ ]:

store.put('responses_emotion',responses_x, format='table', append=True)


# In[ ]:

empty_row={}
for word in all_unique_words:
    empty_row[word]=0


# In[ ]:

response_row=empty_row


# In[ ]:

emotions=response_emotion_df["sentiment"]
y_index=0
for response in cleaned_responses[:5]:
    response_row=empty_row
    for word in all_unique_words:
        response_row[word]+=1
    response_row["emotion_y"]=bag_of_emotions[emotions[y_index]]
    row_df=pd.DataFrame(response_row, index=[y_index])
    store.put('responses_emotion',row_df, format='table', append=True, data_columns=responses_x_cols)
    y_index+=1


# In[ ]:

empty_row=[0]*len(responses_x_cols)


# In[ ]:

emotions=response_emotion_df["sentiment"]
normalized_words=[]
y_index=0
for response in cleaned_responses:
    response_row=empty_row
    index=0
    for word in responses_x_cols:
        if(word in response):
            response_row[index]+=1
        index+=1
    response_row[-1]=bag_of_emotions[emotions[y_index]]
    normalized_words.append(response_row)
    #store.put('responses_emotion',response_row, format='table', append=True, data_columns=responses_x_cols)
    y_index+=1


# In[ ]:

len(normalized_words[0])


# In[ ]:

with open("emotion_matrix.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(normalized_words)


# In[ ]:

responses_x_cols


# In[ ]:

responses_df=pd.read_csv("Donna Files/emo_mat_1.csv", names=responses_x_cols)


# In[18]:

responses_df=pd.read_csv("Donna Files/emo_mat_1.csv")


# In[19]:

responses_df.columns=responses_x_cols


# In[ ]:

responses_df.ix[:,:-1]


# In[ ]:

responses_df["emotion_y"]


# In[ ]:

del(store["responses_emotion"])


# In[ ]:

store=tables.open_file("E:/College Work/CS534 - AI/Donna_h5.h5", mode = "a")


# In[23]:

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import keras.utils


# In[ ]:

store=pd.HDFStore("E:/College Work/CS534 - AI/Donna_h5.h5")


# In[21]:

# split into input (X) and output (Y) variables
X = responses_df.ix[:,:-1]
Y = responses_df["emotion_y"]


# In[39]:

emotions_y=[]
empty_emotion_row=[0]*8
for index in Y:
    emotion_row=empty_emotion_row
    emotion_row[index]=1
    emotions_y.append(emotion_row)


# In[26]:

X_reshaped=X.as_matrix().reshape(9999,36084)


# In[24]:

emotions_y=keras.utils.to_categorical(Y,num_classes=8)


# In[30]:

# create model
model = Sequential()
model.add(Dense(2000, input_dim=X.shape[1], kernel_initializer='uniform', activation='relu'))
model.add(Dense(500, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[31]:

# Fit the model
filepath="Donna Files/emotion_forward_weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_reshaped, emotions_y, epochs=150, batch_size=100, callbacks=callbacks_list)
# calculate predictions


# In[ ]:

predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
