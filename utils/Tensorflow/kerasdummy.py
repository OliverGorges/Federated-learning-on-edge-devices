import tensorflow as tf
import logging 

class Model(tf.keras.Model):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.seq = tf.keras.Sequential([
            tf.keras.Input([320,320,3], 1),
        ])

    def call(self, x):
        x = self.seq(x)
        images, shapes = self.model.preprocess(x)
        prediction_dict = self.model.predict(images, shapes)
        logging.warning('Results')
        logging.info(prediction_dict)
        detections = self.model.postprocess(prediction_dict, shapes)
        boxes = detections['detection_boxes']
        scores = detections['detection_scores'][:,:,None]
        classes = detections['detection_classes'][:,:,None]
        combined = tf.concat([boxes, classes, scores], axis=2)
        return combined