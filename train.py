#定义网络模型并进行训练和保存
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf
import numpy as np

keep_prob=tf.placeholder("float",name='keep_prob')

# convolution and pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

# convolution layer
def lenet5_layer(layer,weight,bias):
    h_conv=conv2d(layer,weight)+bias
    return max_pool_2x2(h_conv)

# connected layer
def dense_layer(layer,weight,bias):
    return tf.matmul(layer,weight)+bias

def main():
    sess=tf.InteractiveSession()
    # parameters
    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1), name="w1")
    b1 =tf.Variable(tf.constant(0.1, shape=[6]) ,name="b1")
    w2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.constant(0.1, shape=[16]) ,name="b2")
    w3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120], stddev=0.1), name="w3")
    b3 = tf.Variable(tf.constant(0.1, shape=[120]) ,name="b3")
    w_fc1 = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1), name="w_fc1")
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[84]) ,name="b_fc1")
    w_fc2 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1), name="w_fc2")
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]) ,name="b_fc2")

    # input layer
    x=tf.placeholder("float",shape=[None,784], name='x')
    y_=tf.placeholder("float",shape=[None,10], name='y_')

    # first layer
    with tf.name_scope('conv1'):
        x_image=tf.pad(tf.reshape(x,[-1,28,28,1]),[[0,0],[2,2],[2,2],[0,0]])
        layer=lenet5_layer(x_image, w1, b1)

    # second layer
    with tf.name_scope('conv2'):
        layer=lenet5_layer(layer, w2, b2)

    # third layer
    with tf.name_scope('conv3'):
        layer=conv2d(layer, w3)+b3
        layer=tf.reshape(layer, [-1, 120])

    # all connected layer
    with tf.name_scope('fc'):
        con_layer=dense_layer(layer, w_fc1, b_fc1)

    # output
    with tf.name_scope('output'):
        con_layer=dense_layer(con_layer, w_fc2, b_fc2)
        y_conv = tf.nn.softmax(tf.nn.dropout(con_layer,keep_prob))
    tf.add_to_collection('network-output', y_conv)

    # train and evalute
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(30000):
        batch=mnist.train.next_batch(50)
        if i%100==0:
            train_accuracy=accuracy.eval(feed_dict={
                x:batch[0],y_:batch[1],keep_prob:1.0
            })
            print("step %d,training accuracy %g"%(i,train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    saver.save(sess, "save/model.ckpt", global_step=30000)
    print("Test accuracy %g"%accuracy.eval(feed_dict={
        x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0
    }))

if __name__=='__main__':
    main()

