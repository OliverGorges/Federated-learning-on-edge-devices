import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import logging

"""
Helper method to modify the ObjecdetectionPipeline files
"""

def setConfig(file, Values, output=None):
    if not output:
        output = file                                                                                                                                                                                                                                         
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()                                                                                                                                                                                                          

    with tf.io.gfile.GFile(file, "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)                                                                                                                                                                                                                 

    pipeline_config.model.ssd.num_classes = 1

    for k, v in Values.items():
        if k == "label_map":
            pipeline_config.train_input_reader.label_map_path = v
            pipeline_config.eval_input_reader[0].label_map_path = v
        elif k == "train_input":
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = v
        elif k == "eval_input":
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = v
        elif k == "num_eval":
            pipeline_config.eval_config.num_examples = int(v) 
        elif k == "num_classes":
            pipeline_config.model.ssd.num_classes = int(v)                                                                                                                                                                                   
        elif k == "checkpoint":
            pipeline_config.train_config.fine_tune_checkpoint = v
        else:
            print(f'Key: {k} not found')

    config_text = text_format.MessageToString(pipeline_config)  
    #print(config_text)                                                                                                                                                                                                      
    with tf.io.gfile.GFile(output, "wb") as f:                                                                                                                                                                                                                       
        f.write(config_text)                                                                                                                                                                                                                                          


