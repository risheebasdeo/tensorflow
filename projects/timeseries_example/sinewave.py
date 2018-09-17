import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# list object containing time series data
# type = list
data = pd.read_csv('~/tensorflow/scripts/tutorial/sinewave.csv', header=None)
output_values = data.values
data = data.values

# time steps to process
time_steps = 24

# iterations
epochs = 50

# number of time steps for the final prediction
prediction_time_steps = 48

# learning rate
learning_rate = 0.1

# number of features (i.e. stock price, trading volume, boolean representing significant news)
features = 1

# number of nodes in a LSTM cell
nodes = 15

# number of layers (each containing one LSTM cell)
lstm_layers = 1

# output dimension (the stock price normalized)
outputs = 1

# tensor
x_tensor = tf.placeholder(tf.float32, [None, time_steps, features], name='feature')
y_tensor = tf.placeholder(tf.float32, [None, time_steps, outputs], name='label')

# construct the lstm (rnn)
layers = [tf.contrib.rnn.BasicLSTMCell(num_units=nodes) for layer in range(lstm_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, x_tensor, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, nodes], name='rnn')
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, outputs)
predictions = tf.reshape(stacked_outputs, [-1, time_steps, outputs])

# optimize the model by calculating the loss
loss = tf.reduce_mean(tf.square(predictions - y_tensor))
tf.summary.scalar('loss',loss)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

no_of_batches = int(len(data)) / time_steps
print "total extracted data points: ", len(data)
print "data points per epoch: ", no_of_batches * time_steps
print "batches per epoch: ", no_of_batches
print "batch size: ", time_steps

# Train the model
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
    init.run()
    plt.figure(figsize=(15, 6))
    #plt.axis([0, 70, -1, 1])
    plt.ion()
    epoch_prediction = []
    for i in range(epochs):
        # plotting the target values array
        plt.plot((output_values.reshape(-1,)), c='b', label="target", linewidth=6.0, zorder=5)

        batch_time_frame_prediction = [None] * (len(output_values))

        # remove the previous batch's prediction
        # this allows only the latest prediction to be displayed
        if i != 0:
            epoch_prediction.pop().pop().remove()

        window_start = 0
        window_end = time_steps + window_start

        # stop the window 1 slot before the last element in the array
        # since we don't have a value to compare the last elements prediction to
        while window_end < len(output_values):
            x_time_steps = data[window_start:window_end].reshape(1, time_steps, features)

            # numpy array [1][time_steps][1]
            y_time_steps = output_values[window_start+1:window_end+1].reshape(1, time_steps, outputs)

            # train the model (optimize after calculating the loss)
            sess.run(training_op, feed_dict={x_tensor: x_time_steps, y_tensor: y_time_steps})

            # get the loss for the trained batch
            mse = sess.run(loss, feed_dict={x_tensor: x_time_steps, y_tensor: y_time_steps})
            y_prediction = sess.run(predictions, feed_dict={x_tensor: x_time_steps})

            # build the prediction line for the current batch
            for y_index, y_val in enumerate(y_prediction.reshape(-1,)):
                batch_time_frame_prediction[y_index+window_start+1] = y_val

            print("Epoch: " + str(i+1), " Batch: " + str(window_start+1), "Error: " + str(mse))

            window_start += 1
            window_end = time_steps + window_start

        # plot the prediction for the latest batch
        epoch_prediction.append(plt.plot(batch_time_frame_prediction, c='r', label="prediction", linewidth=4.0, zorder=10))
        plt.pause(.00001)

    x_time_step_forecast = data[len(data)-time_steps:len(data)]
    # plot the final prediction starting with the original data points
    final_prediction = output_values.reshape(-1)

    for p in range(prediction_time_steps):
        x_time_step_forecast = x_time_step_forecast.reshape(1, time_steps, features)
        print "prediction input: ", x_time_step_forecast.reshape(-1,)
        y_forecast = predictions.eval(feed_dict={x_tensor: x_time_step_forecast})
        print "prediction output: ", y_forecast.reshape(-1,)

        # predict the next value
        # remove the first element of the array and append the latest predicted value
        # the first 3 elements represents the 3 input features
        x_time_step_forecast = np.delete(x_time_step_forecast,[0])
        x_time_step_forecast = np.append(x_time_step_forecast, y_forecast[0][0][0])

        # plot the predicted value
        final_prediction = np.append(final_prediction,y_forecast[0][0][0])
        plt.plot(final_prediction, c="g", linewidth=2.0, zorder=0)
        plt.pause(.00001)
        print final_prediction.reshape(-1,)

    saver.save(sess, "./sine_wave")

while True:
    plt.pause(1)
