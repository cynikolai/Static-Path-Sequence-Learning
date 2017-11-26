import numpy as np
import tensorflow as tf
from random import randint
import datetime

maxSeqLength = 20
numDimensions = 68
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 50000

data = np.load('data.npy')


def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength, numDimensions])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = data[num-1:num]
    return arr, labels


def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength, numDimensions])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = data[num-1:num]
    return arr, labels

tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
data = tf.placeholder(tf.float32, [batchSize, maxSeqLength, numDimensions])

lstmCell = tf.contrib.rnn.GRUCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

softmax = tf.nn.softmax(prediction)
normalized = tf.nn.l2_normalize(softmax, 1)
confidence = tf.reduce_max(normalized, 1)
predictedLabels = tf.argmax(prediction, 1)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
precision = tf.metrics.precision(labels, predictedLabels)
recall = tf.metrics.recall(labels, predictedLabels)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                      logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir += "/"
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(iterations):
    nextBatch, nextBatchLabels = getTrainBatch()
    sess.run(optimizer, {data: nextBatch, labels: nextBatchLabels})

    if(i % 50 == 0):
        summary = sess.run(merged, {data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt",
                               global_step=i)
        print("saved to %s" % save_path)
writer.close()

iterations = 10
accuracy_average = 0
precision_average = 0
recall_average = 0
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch()
    accuracy_average += (sess.run(accuracy, {data: nextBatch,
                                             labels: nextBatchLabels}))
    precision_average += (sess.run(precision, {data: nextBatch,
                                               labels: nextBatchLabels}))
    recall_average += (sess.run(recall, {data: nextBatch,
                                         labels: nextBatchLabels}))
print "Accuracy: " + str(accuracy_average / iterations)
precision_val = precision_average / iterations
recall_val = recall_average / iterations
f_score = 2 / ((1 / precision_val) + (1 / recall_val))
print "F-Score: " + str(f_score)