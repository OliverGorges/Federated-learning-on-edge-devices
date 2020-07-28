import tensorflow as tf
import os
from google.protobuf import text_format
from object_detection import export_tflite_ssd_graph_lib
from object_detection.protos import pipeline_pb2

from object_detection.utils import config_util
from object_detection.builders import model_builder
import numpy as np

"""
Converter form trained checkpoints/graph to a tflite graph

Original Commands:
tflite_convert --graph_def_file Federated-learning-on-edge-devices/Traindata/model/TFlite/tflite_graph.pb --output_file=./detect.tflite --output_format=TFLITE --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --allow_custom_ops

TODO: Problem with Conversion, results not usable
"""


def convertModel(input_dir, output_dir, pipeline_config="", checkpoint:int=-1, ):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    files = os.listdir(input_dir)
    if pipeline_config == "":
        pipeline_config = [pipe for pipe in files if pipe.endswith(".config")][0]
    pipeline_config_path = os.path.join(input_dir, pipeline_config)


    # Find latest or given checkpoint
    checkpoint_file = ""
    checkpointDir = os.path.join(input_dir, 'checkpoint')
    for chck in sorted(os.listdir(checkpointDir)):
        if chck.endswith(".index"):
            checkpoint_file = chck[:-6]
            # Stop search when the requested was found
            if chck.endswith(str(checkpoint)): 
                break
    print("#####################################")
    print(checkpoint_file)
    print("#####################################")
    #ckeckpint_file = [chck for chck in files if chck.endswith(f"{checkpoint}.meta")][0]
    trained_checkpoint_prefix = os.path.join(checkpointDir, checkpoint_file)


    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    detection_model = model_builder.build(configs['model'], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(trained_checkpoint_prefix).expect_partial()


    class MyModel(tf.keras.Model):
        def __init__(self, model):
            super(MyModel, self).__init__()
            self.model = model
            self.seq = tf.keras.Sequential([
                tf.keras.Input([320,320,3], 1),
            ])

        def call(self, x):
            x = self.seq(x)
            images, shapes = self.model.preprocess(x)
            prediction_dict = self.model.predict(images, shapes)
            detections = self.model.postprocess(prediction_dict, shapes)
            boxes = detections['detection_boxes']
            scores = detections['detection_scores'][:,:,None]
            classes = detections['detection_classes'][:,:,None]
            combined = tf.concat([boxes, classes, scores], axis=2)
            return combined

    km = MyModel(detection_model)

    y = km.predict(np.random.random((1,320,320,3)).astype(np.float32))
    print(y)
    converter = tf.lite.TFLiteConverter.from_keras_model(km)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_new_converter = True
    converter.allow_custom_ops = False
    tflite_model = converter.convert()

    open(os.path.join(output_dir, 'model.tflite'), 'wb').write(tflite_model)


    """
    # Load config
    files = os.listdir(input_dir)
    if pipeline_config == "":
        pipeline_config = [pipe for pipe in files if pipe.endswith(".config")][0]
    pipeline_config_path = os.path.join(input_dir, pipeline_config)

    # Find latest or given checkpoint
    checkpoint_file = ""
    for chck in sorted(files):
        if chck.endswith(".meta"):
            checkpoint_file = chck[:-5]
            # Stop search when the requested was found
            if chck.endswith(str(checkpoint)): 
                break
    print(checkpoint_file)
    #ckeckpint_file = [chck for chck in files if chck.endswith(f"{checkpoint}.meta")][0]
    trained_checkpoint_prefix = os.path.join(input_dir, checkpoint_file)



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export frozengraph from checkpoint
    export_tflite_ssd_graph(pipeline_config_path, trained_checkpoint_prefix, output_dir, True, 20, 1, 100)

    # TFLite I/O
    input_array = ["normalized_input_image_tensor"]
    output_array = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
    
    # Convert frozengraph to tflite
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        os.path.join(output_dir, "tflite_graph.pb"), 
        input_array, 
        output_array,
        input_shapes={'normalized_input_image_tensor':[1, 300, 300, 3]}
        )

    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.add_postprocessing_op = True
    converter.experimental_new_converter =True
    converter.target_spec.supported_op = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(os.path.join(output_dir, "converted_model.tflite"), "wb").write(tflite_model)
    """



def export_tflite_ssd_graph(pipeline_config_path, trained_checkpoint_prefix, output_directory, use_regular_nms=False, max_classes_per_detection=2, max_detections=10, add_postprocessing_op=True ):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    #text_format.Merge(FLAGS.config_override, pipeline_config)
    export_tflite_ssd_graph_lib.export_tflite_graph(
        pipeline_config, trained_checkpoint_prefix, output_directory,
        add_postprocessing_op, max_detections,
        max_classes_per_detection, use_regular_nms)


def convert_from_saved_model(inputDir, outputDir):
    """
    deprecated: Replaced with convertModel
    """
    savedModel = os.path.join(inputDir, "saved_model")
    print(savedModel)
    converter =  tf.compat.v1.lite.TFLiteConverter.from_saved_model(
        savedModel, 
        input_shapes={'image_tensor':[1, 300, 300, 3]}
        )
    converter.add_postprocessing_op = True
    tfliteModel = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile(os.path.join(outputDir, "converted_model.tflite"), 'wb') as f:
        f.write(tfliteModel)


if __name__ == "__main__":
    
    inputDir = os.path.join("Traindata", "model", "MaskDetect20k")
    outputDir = os.path.join("Traindata", "model", "TFLite")
    convertModel(inputDir, outputDir, checkpoint=18431)
    #convert_from_saved_model(inputDir, outputDir)