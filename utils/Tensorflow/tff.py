"""
Script that loads the values from a Checkpoint
"""

import sys
import tensorflow as tf
import os
import numpy as np
import random 
import time
import mpipe 

def aggregateVariables(checkpoint_dicts):
    chkps = checkpoint_dicts
    aggregatedData = {}
    for key in chkps[0].keys():
        print(key)
        if key == "id":
            continue
        inputData = [x[key][0] for x in chkps]
        print(inputData)
        aggregatedData[key] = np.average([x[key][0] for x in chkps], axis=0, weights=[1 for x in chkps])
        print(aggregatedData[key])
    return aggregatedData
    
def writeCheckpointValues(data, ref, out):
    tf.reset_default_graph()
    if isinstance(ref, str):
        graph = tf.compat.v1.train.import_meta_graph(model_dir) 
    else:
        graph = ref
    with tf.Session() as sess:
        graph.as_default()
        tf.compat.v1.global_variables_initializer().run()
        var = tf.compat.v1.trainable_variables()
        for v in var:
            try:
                variable = v.value().eval(session=sess)
                variable.load(data[v.value().name], sess)
            except:
                print(f"Missing Key {v.value().name}")
        saver = tf.compat.v1.train.Saver(var)
        saved_path = saver.save(sess, out)
        print(saved_path)
        sess.close()

def readCheckpointValues(path, trainable=True):
    model_dir = path[0]
    id = path[1]
    tf.reset_default_graph()
    print(f"Load Checkpoint {model_dir} {id}")
    # Read Checkpoint
    #sess = tf.compat.v1.InteractiveSession()
    graph = tf.compat.v1.train.import_meta_graph(model_dir) 

    #g = tf.compat.v1.get_default_graph()

    with tf.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        if trainable:
            var = tf.compat.v1.trainable_variables()
        else:
            var = tf.compat.v1.global_variables()
        values = {"id": id}
        randomMulti = random.random() + 0.5
        #Modify values
        for v in var:
            data = v.value().eval(session=sess)
            data = data * randomMulti
            values[str(v.value().name)] = (data, v.get_shape().as_list())
        sess.close()
        return values

def sendData(path, dest, endpoint, trainable=True):
    pass

if __name__ == "__main__":
    
    model_id = "FaceDetect"
    model_dir = os.path.join("Traindata", "model", model_id, "model.ckpt-4990.meta")
    checkpoints = [model_dir for x in range(2)]

    print(checkpoints)
    #stage1 = mpipe.UnorderedStage(readCheckpointValues, len(checkpoints))
    #pipe = mpipe.Pipeline(stage1)
    chkps = []
    aggregatedData = {}
    t1 = time.time()
    #chkps.append(readCheckpointValues(checkpoints[0], "Ref", False)) # load Reference Checkpoint with all Keys
    t2 = time.time()
    print(f'Load Ref: {t2-t1}')
    for i, c in enumerate(checkpoints):
        #pipe.put((c, i))
        chkps.append(readCheckpointValues((c,i)))
    #pipe.put(None)
    #for result in pipe.results():
        #chkps.append(result)
    t3 = time.time()
    print(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}')
    aggregatedData = aggregateVariables(chkps)
    t4 = time.time()
    print(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}')

    writeCheckpointValues(aggregatedData, model_dir, os.path.join("Traindata", "model", "graphmod", "modi_model.ckpt" ))
    t5 = time.time()
    print(f'Load Ref: {t2-t1}, Load Checkpoints: {t3-t2}, Modify Key: {t4-t3}, Save Checkpoint: {t5-t4}')
