import pandas as pd
import numpy as np
import tensorflow as tf
import math

bc_df=pd.read_csv("/home/kavin/Silo/College Work/AI/DataSets/brown_tag_lines.txt",sep="_", names=("word","pos_tag"))
unique_words=pd.Series(bc_df["word"]).unique()

bag_of_words={}
index=0.0
for word in unique_words:
    bag_of_words[word]=index
    index+=1


unique_tags=bc_df["pos_tag"].unique()
bag_of_tags={}
index=0.0
for tag in unique_tags:
    bag_of_tags[tag]=index
    #index+=1

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 10
display_step = 1

# Network Parameters
n_hidden_1 = 10000 # 1st layer number of features
n_hidden_2 = 500 # 2nd layer number of features
n_input = 53023 # MNIST data input (img shape: 28*28)
n_classes = 313 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

split_index=int(math.floor(len(bc_df)*0.75))
split_index
train, test = bc_df[:split_index], bc_df[split_index:]

def getBatchData(df, index):
    till_index=index+10
    if(len(df)<till_index):
        till_index=len(df)
    subset=df[index:index+10]
    df_x, df_y=[], []
    for record in subset.itertuples():
        tag_flags=bag_of_tags
        word_flags=bag_of_words
        word_flags[record[1]]=1
        tag_flags[record[2]]=1
        df_x.append(word_flags.values())
        df_y.append(tag_flags.values())
    return np.asarray(df_x), np.asarray(df_y)

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
  # Restore variables from disk.
    sess.run(init)
    train_ds=train
    # Training cycle
    for epoch in range(training_epochs):
	print epoch
        avg_cost = 0.
        total_batch = len(train_ds)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = getBatchData(train_ds,i)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
        save_path = saver.save(sess, "/home/kavin/Silo/College Work/AI/tagger_model.ckpt")
        print("Model saved in file: %s" % save_path)
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    print correct_prediction
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    df_x, df_y=getTestData(test)
    print("Accuracy:", accuracy.eval({x: df_x, y: df_y}))

def getTestData(df):
    df_x, df_y=[], []
    for record in df.itertuples():
        tag_flags=bag_of_tags
        word_flags=bag_of_words
        word_flags[record[1]]=1
        tag_flags[record[2]]=1
        df_x.append(word_flags.values())
        df_y.append(tag_flags.values())
    return np.asarray(df_x), np.asarray(df_y)

def getAccuracy(session):
    session.as_default()
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    print correct_prediction
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    df_x, df_y=getTestData(test[:100])
    print("Accuracy:", accuracy.eval({x: df_x, y: df_y}))
