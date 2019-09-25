#将卷积层的weight加上小噪声，计算网络在测试集上的准确率
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def noise(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.05))

if __name__ == '__main__':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./save/model.ckpt-30000.meta')  #加载网络结构图
        saver.restore(sess, './save/model.ckpt-30000')      #加载网络参数

        pred = tf.get_collection('network-output')[0]
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('x').outputs[0]
        y_ = graph.get_operation_by_name('y_').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
#-------------------------------------------------------------------------------------------------------------------------
        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        w3 = graph.get_tensor_by_name("w3:0")

        w1_noise=noise(w1.shape)
        w2_noise=noise(w2.shape)
        w3_noise=noise(w3.shape)
        w1_update = tf.assign(w1, w1 + w1_noise)
        w2_update = tf.assign(w2, w2 + w2_noise)
        w3_update = tf.assign(w3, w3 + w3_noise)
        sess.run(tf.variables_initializer([w1_noise,w2_noise,w3_noise]))

        sess.run(w1_update)
        sess.run(w2_update)
        sess.run(w3_update)
#--------------------------------------------------------------------------------------------------------------------------
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



