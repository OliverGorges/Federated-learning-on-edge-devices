import tensorflow as tf
import os
from google.protobuf import text_format
from object_detection import export_tflite_ssd_graph_lib
from object_detection.protos import pipeline_pb2


def convertModel(input_dir, output_dir, pipeline_config="", checkpoint:int=-1, ):

    files = os.listdir(input_dir)
    if pipeline_config == "":
        pipeline_config = [pipe for pipe in files if pipe.endswith(".config")][0]
    pipeline_config_path = os.path.join(input_dir, pipeline_config)

    ckeckpint_file = ""
    for chck in sorted(files):
        if chck.endswith(".meta"):
            ckeckpint_file = chck[:-5]
            # Stop search when the requested was found
            if chck.endswith(str(checkpoint)): 
                break

    #ckeckpint_file = [chck for chck in files if chck.endswith(f"{checkpoint}.meta")][0]
    trained_checkpoint_prefix = os.path.join(input_dir, ckeckpint_file)



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    export_tflite_ssd_graph(pipeline_config_path, trained_checkpoint_prefix, output_dir)

    input_array = ["normalized_input_image_tensor"]
    output_array = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
    
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        os.path.join(output_dir, "tflite_graph.pb"), 
        input_array, 
        output_array,
        input_shapes={'normalized_input_image_tensor':[1, 300, 300, 3]}
        )

    converter.allow_custom_ops = True
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.add_postprocessing_op = True
    tflite_model = converter.convert()
    open(os.path.join(output_dir, "converted_model.tflite"), "wb").write(tflite_model)

"""
    #Prepare Model for TFlite
    tflite_convert --graph_def_file Federated-learning-on-edge-devices/Traindata/model/TFlite/tflite_graph.pb --output_file=./detect.tflite --output_format=TFLITE --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --allow_custom_ops
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

if __name__ == "__main__":
    
    inputDir = os.path.join("Traindata", "model", "MaskDetect20k")
    outputDir = os.path.join("Traindata", "model", "TFLite")
    convertModel(inputDir, outputDir, checkpoint=18431)