
import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.preprocessing.image import img_to_array
from utils.Tensorflow.models import simpnet, sampleModel
import time
import json

from matplotlib import pyplot as plt

NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100
USERS = 10
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        else:
            return json.JSONEncoder.default(self, obj)

mnist_train, mnist_test = tf.keras.datasets.cifar10.load_data()

def print_samples(dataset):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(dataset[i][-1]['x'][-1].reshape(32, 32, 3))
        plt.title(f"Class: {(dataset[i][-1]['y'][-1])}")
        plt.axis("off")
        print(dataset[i][-1]['y'][-1])
    plt.show()

def preprocessFed(source, output):
  output_sequence = []
  all_samples = [i for i, d in enumerate(source[1]) if d == output]
  for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
    batch_samples = all_samples[i:i + BATCH_SIZE]
    output_sequence.append({
        'x':
            np.array([source[0][i] / 255.0 for i in batch_samples],
                     dtype=np.float32),
        'y':
            np.array([source[1][i][0] for i in batch_samples], dtype=np.int32)
    })
  return output_sequence


federated_train_data = [preprocessFed(mnist_train, d) for d in range(USERS)]
federated_test_data = [preprocessFed(mnist_test, d) for d in range(USERS)]

def preprocessDL(source, split):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1])]
    if split == 1:
        sub_samples = [all_samples]
    else:
        print(len(all_samples))
        s = int(len(all_samples) * split)
        print(s)
        sub_samples = [all_samples[:s], all_samples[s:]]
    for s in range(len(sub_samples)):
        output_sequence.append((
            np.array([source[0][i] / 255.0 for i in sub_samples[s]], dtype=np.float32),
            np.array([source[1][i][0] for i in sub_samples[s]], dtype=np.int32)
        ))
    return output_sequence

deeplearning_train_val_data = preprocessDL(mnist_train, 0.8)
deeplearning_test_data = preprocessDL(mnist_test, 1)

#print_samples()

BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None], dtype=tf.int32))
BATCH_TYPE = tff.to_type(BATCH_SPEC)

print(str(BATCH_TYPE))

MODEL_SPEC = collections.OrderedDict(
    weights=tf.TensorSpec(shape=[32, 32, 3, 10], dtype=tf.float32),
    bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
MODEL_TYPE = tff.to_type(MODEL_SPEC)

print(MODEL_TYPE)

def create_keras_model():
    

    return simpnet.get_model()

def runFL(runs, test_break, log):
    t1= time.time()
    info = {}
    def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=BATCH_SPEC,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


    state = iterative_process.initialize()

    state, metrics = iterative_process.next(state, federated_train_data)
    print('round  1, metrics={}'.format(metrics))

    NUM_ROUNDS = int(runs / test_break)
    results = []
    metrics = []
    model = create_keras_model()
    for i in range(test_break):
        for round_num in range(2, NUM_ROUNDS):
            state, m = iterative_process.next(state, federated_train_data)
            print('round {:2d}, metrics={}'.format(round_num, m))
            metrics.append(m)
        tff.learning.assign_weights_to_keras_model(model, state.model)
        results.append(runTest(model, NUM_ROUNDS*(i + 1), 'FL'))
    print(f'FL: {time.time() - t1}')
    info['rounds'] = results
    info['metrics'] = metrics
    info['time'] = time.time() - t1
    log['FL'] = info
    return log

def runDL(runs, test_break, log):
    t1= time.time()
    info = {}
    model = create_keras_model()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer='sgd',
            metrics=['accuracy'])
    
    epochs= int(runs / test_break)
    results = []
    metrics = []
    for i in range(test_break):
        history = model.fit(
            deeplearning_train_val_data[0][0],
            deeplearning_train_val_data[0][1],
            validation_data=deeplearning_train_val_data[1],
            epochs=epochs,
            batch_size=100,
            steps_per_epoch=10
            )
        results.append(runTest(model, epochs*(i + 1), 'DL'))
        metrics.append(history.history)
    print(f'DL: {time.time() - t1}')
    info['rounds'] = results
    info['metrics'] = metrics
    info['time'] = time.time() - t1
    log['DL'] = info
    return log

def initialTest(log):
    model = create_keras_model()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer='sgd',
            metrics=['accuracy'])
    log['init'] = runTest(model, 0, 'init')
    return log

def runTest(model, step, method):
    tp = {}
    general_tp = 0
    tn = {}
    general_tn = 0
    y_pred = []
    for val, label in zip(*deeplearning_test_data[0]):
        img_array = img_to_array(val)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        y_pred.append(score)
        predicted_class = int(np.argmax(score))
        label = int(label)
        if predicted_class == label:
            general_tp += 1
            if predicted_class in tp:
                tp[predicted_class] += 1
            else:
                tp[predicted_class] = 1
        else:
            general_tn += 1
            if label in tn:
                if predicted_class in tn[label]:
                    tn[label][predicted_class] += 1
                else:
                    tn[label][predicted_class] = 1
            else:
                tn[label] = {}
                tn[label][predicted_class] = 1
        if general_tp + general_tn >= 1000:
            break
        #print("This image most likely belongs to {} with a {:.2f} percent confidence. Truth: {}".format(np.argmax(score), 100 * np.max(score), label))
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = scce(deeplearning_test_data[0][1][:1000], y_pred).numpy()
    print(f'{method} {step}: Test Loss: {loss}, TP {general_tp}/{general_tn} TN')
    return {'step': step, 'loss': loss, 'tp': tp, 'tn': tn, 'general_tp': general_tp, 'general_tn': general_tn}

if __name__ == "__main__":  
    log = {}
    r = 1
    test = 1
    log['info'] = {'time': time.time(), 'rounds': r, 'testruns': test, 'classes': 10}
    log = initialTest(log)
    log = runFL(r, test, log)
    log = runDL(r, test, log)
    print(log)
    with open("metrics.json", 'w') as outfile:
        json.dump(log, outfile,cls=NumpyEncoder)


