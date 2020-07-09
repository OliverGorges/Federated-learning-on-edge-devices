"""
Script that loads the values from a Checkpoint
"""

import sys
import tensorflow as tf
import os

model_id = "FaceDetect"
model_dir = os.path.join("TrainData", "graphmod", "model.ckpt-5081")
out =  os.path.join("TrainData", "graphmod", "modi_model.ckpt")


print("Load Modded Checkpoint")
# Read Checkpoint
#sess = tf.compat.v1.InteractiveSession()

if os.path.exists(out+".meta"):
    print("rerun")
    graph = tf.compat.v1.train.import_meta_graph(out+".meta")   
else:
    print("new run")
    graph = tf.compat.v1.train.import_meta_graph(model_dir+".meta") 


#g = tf.compat.v1.get_default_graph()

with tf.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    var = tf.compat.v1.trainable_variables()
    
    #Modify values
    v = var[1]
    print(v.value())
    data = v.value().eval(session=sess)
    print(data)
    ndata = data * 2
    print(ndata)

    #assign_op = tf.compat.v1.assign(v, ndata)
    #sess.run(assign_op)
    v.load(ndata, sess)

    test =  tf.compat.v1.trainable_variables()
    
    print(test[1].value().eval())

    saver = tf.compat.v1.train.Saver(var)
    saved_path = saver.save(sess, out)
    print(saved_path)


