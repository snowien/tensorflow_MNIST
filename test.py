#加载保存的模型并在测试集上进行测试
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


if __name__=='__main__':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./save/model.ckpt-30000.meta')
        saver.restore(sess, './save/model.ckpt-30000')

        pred = tf.get_collection('network-output')[0]
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('x').outputs[0]
        y_ = graph.get_operation_by_name('y_').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#-------------------------------------------------------------------------------------------------------------------------
        '''
        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        w3 = graph.get_tensor_by_name("w3:0")
        print(sess.run(w1))
        print("aaaaaaaaaaaaaaaaa")
        print(sess.run(w2))
        print("aaaaaaaaaaaaaaaaa")
        print(sess.run(w3))
        '''