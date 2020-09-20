from object_detection.dataset_tools.create_coco_tf_record import _create_tf_record_from_coco_annotations
from utils.Dataset.cocoAnnotationConverter import  XmlConverter, JsonConverter
from utils.Dataset import pipeline_config
import os
from shutil import copy
import pathlib
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
import time
import logging
import cv2
import numpy as np
from object_detection import model_hparams
from object_detection import model_lib
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2

from object_detection.utils import config_util
from object_detection.builders import model_builder
import numpy

logging.basicConfig(level=logging.DEBUG)

def splitList(data, i):
    rest = []
    if i > len(data):
        return [data]
    while not len(data)%i == 0:
        rest.append(data.pop())
    result = numpy.split(numpy.array(data), i)
    result[i-1] = numpy.concatenate((result[i-1], rest))
    logging.debug([len(s) for s in result])
    return result


def augmentData(imageDir, annotationDir, outputDir, split=1):

    if split > 1:
        images = splitList(sorted(os.listdir(imageDir)), split)
        annos = splitList(sorted(os.listdir(annotationDir)), split)
        imageSubsets = []
        annotationSubsets = []
        for i in range(split):
            folder = os.path.join(outputDir, f'Subset{i}')
            if not os.path.exists(folder):
                os.mkdir(folder)
                os.mkdir(os.path.join(folder, 'images'))
                os.mkdir(os.path.join(folder, 'annotations'))

        for i, imageset in enumerate(images):
            folder = os.path.join(outputDir, f'Subset{i}', 'images')
            for img in imageset:
                copy(os.path.join(imageDir,img), folder)
            imageSubsets.append(folder)
        for i, annoset in enumerate(annos):
            folder = os.path.join(outputDir, f'Subset{i}', 'annotations')
            if not os.path.exists(folder):
                os.mkdir(folder)
            for anno in annoset:
                copy(os.path.join(annotationDir, anno), folder)
            annotationSubsets.append(folder)
        return imageSubsets, annotationSubsets
    else:
        imgOut = os.path.join(outputDir, "images")
        annoOut = os.path.join(outputDir, "annotations")
        for img in sorted(os.listdir(imageDir)):
            copy(os.path.join(imageDir,img), imgOut)
        for anno in sorted(os.listdir(annotationDir)):
            copy(os.path.join(annotationDir, anno), annoOut)
        return [imgOut], [annoOut]

def prepareTFrecord(dataDir, annoDir, outputDir, labelmap=None, annoFormat=None, split=0.7):
    config = {}

    # Convert single annotation Files to AnnotationFile for Train and Eval
    if annoFormat == "XML":
        annotationFiles, size, classes =  XmlConverter().convert(dataDir, annoDir,  outputDir, labelmap, split)
    elif annoFormat == "JSON":
        annotationFiles, size, classes =  JsonConverter().convert(dataDir, annoDir,  outputDir, labelmap, split)
    else:
        annotationFiles =  [file for file in os.listdir(outputDir) if file.endswith(".json")]
        size = [0, 10]
    print( annotationFiles)


    config['num_classes'] = classes

    # Creates TFrecord for Train and Eval Annotations
    for annotationFile in annotationFiles:
        name = str(os.path.basename(annotationFile)).split(".")[0]
        if "Train" in name:
            record = os.path.join(outputDir, 'Train.record')
            _create_tf_record_from_coco_annotations(annotationFile, dataDir, record, False, 1)
            config["train_input"] = str(pathlib.Path(record).absolute())+"-00000-of-00001"
        elif "Eval" in name:
            record = os.path.join(outputDir, 'Eval.record')
            _create_tf_record_from_coco_annotations(annotationFile, dataDir, record, False, 1)
            config["eval_input"] = str(pathlib.Path(record).absolute())+"-00000-of-00001"
            config["num_eval"] = size[1]

    return config

def exportFrozenGraph(modelDir, input_shape=None ):
    pipeline_config_path=""
    trained_checkpoint_prefix = ""
    trained_checkpoint = os.path.join(modelDir, "checkpoint")
    for f in sorted(os.listdir(modelDir)):
        if f.endswith(".config"):
            pipeline_config_path = os.path.join(modelDir, f)
    for f in sorted(os.listdir(trained_checkpoint)):
        if f.endswith(".index"):
            trained_checkpoint_prefix = os.path.join(trained_checkpoint, f)[:-6]
    if not input_shape:
        input_shape = [None, 320, 320, 3]

    print(trained_checkpoint_prefix)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    outputDir = os.path.join(modelDir, "output")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    exporter_lib_v2.export_inference_graph(
        "float_image_tensor", pipeline_config, trained_checkpoint,  outputDir)

def train_eval( modelOutput, dataDir, tfRecordsConfig=None, model="ssd_mobilenet_v2_coco_2018_03_29", steps=1000, num_workers=1, eval_every_n_steps=1000):
    modelDir = os.path.join("Traindata", "model", model)

    if  eval_every_n_steps > steps:
        eval_every_n_steps = steps

    if not steps%eval_every_n_steps == 0:
        logging.warning(f"Can't run all steps with current eval step size, {steps%eval_every_n_steps} will be missing!")

    if not os.path.exists(modelOutput):
        os.makedirs(modelOutput)

    # Create Pipeline Config
    config = {}

    # Load TFRecords
    if tfRecordsConfig is None:
        search = os.listdir(dataDir)
        for f in search:
            if f.startswith('Train.record'):
                config["train_input"] = str(pathlib.Path(os.path.join(dataDir, f)).absolute())
            if f.startswith('Eval.record'):
                config["eval_input"] = str(pathlib.Path(os.path.join(dataDir, f)).absolute())
                config["num_eval"] = 10
    else:
        if "train_input" in tfRecordsConfig:
            config["train_input"] = tfRecordsConfig["train_input"]

        if "eval_input" in tfRecordsConfig:
            config["eval_input"] = tfRecordsConfig["eval_input"]
            config["num_eval"] = tfRecordsConfig["num_eval"]
    config["label_map"] = str(pathlib.Path(os.path.join(modelDir, "labelmap.pbtxt")).absolute())

    # Check if there are checkpoints from older runs
    checkpoint = None
    for f in os.listdir(modelOutput):
        if f.endswith(".index"):
            checkpoint = os.path.join(modelOutput, f)[:-6]

    if checkpoint is None:
        finetune_checkpoint = ""
        for f in os.listdir(os.path.join(modelDir, "checkpoint")):
            if f.endswith(".index"):
               finetune_checkpoint = f[:-6]
        config["checkpoint"] = str(pathlib.Path(os.path.join(modelDir, "checkpoint", finetune_checkpoint)).absolute())
    else:
        config["checkpoint"] = str(pathlib.Path(checkpoint).absolute())


    config['num_classes'] = tfRecordsConfig['num_classes']


    #pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
    pipeline_config_path = os.path.join(modelOutput, "custom_pipeline.config")
    # Modify Pipeline file with the configuration data
    pipeline_config.setConfig(os.path.join(modelDir, "pipeline.config"), config, pipeline_config_path)




    # TF2
    tf.keras.backend.clear_session()


    tf.config.set_soft_device_placement(True)

    for train_steps in range(eval_every_n_steps, steps+1, eval_every_n_steps):
        # train Model
        if num_workers > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()

        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=pipeline_config_path,
                model_dir=modelOutput,
                train_steps=train_steps,
                checkpoint_every_n=eval_every_n_steps,
                checkpoint_max_to_keep=2)

        logging.info(f"Trainsteps: {train_steps}")
        # Run Eval once and then let it timeout to containue with training
        model_lib_v2.eval_continuously(
            pipeline_config_path= pipeline_config_path,
            model_dir= modelOutput,
            train_steps= train_steps,
            sample_1_of_n_eval_examples= 1,
            sample_1_of_n_eval_on_train_examples=(1),
            checkpoint_dir=modelOutput,
            wait_interval=10, timeout=10)

def train( modelOutput, dataDir, tfRecordsConfig=None, model="ssd_mobilenet_v2_coco_2018_03_29", steps=1000, num_workers=1):
    modelDir = os.path.join("Traindata", "model", model)

    if not os.path.exists(modelOutput):
        os.makedirs(modelOutput)

    # Create Pipeline Config
    config = {}

    # Load TFRecords
    if tfRecordsConfig is None:
        search = os.listdir(dataDir)
        for f in search:
            if f.startswith('Train.record'):
                config["train_input"] = str(pathlib.Path(os.path.join(dataDir, f)).absolute())
            if f.startswith('Eval.record'):
                config["eval_input"] = str(pathlib.Path(os.path.join(dataDir, f)).absolute())
                config["num_eval"] = 10
    else:
        if "train_input" in tfRecordsConfig:
            config["train_input"] = tfRecordsConfig["train_input"]

        if "eval_input" in tfRecordsConfig:
            config["eval_input"] = tfRecordsConfig["eval_input"]
            config["num_eval"] = tfRecordsConfig["num_eval"]
    config["label_map"] = str(pathlib.Path(os.path.join(dataDir, "labelmap.pbtxt")).absolute())

    # Check if there are checkpoints from older runs
    checkpoint = None
    for f in os.listdir(modelOutput):
        if f.endswith(".index"):
            checkpoint = os.path.join(modelOutput, f)[:-6]

    if checkpoint is None:
        finetune_checkpoint = ""
        for f in os.listdir(os.path.join(modelDir, "checkpoint")):
            if f.endswith(".index"):
               finetune_checkpoint = f[:-6]
        config["checkpoint"] = str(pathlib.Path(os.path.join(modelDir, "checkpoint", finetune_checkpoint)).absolute())
    else:
        config["checkpoint"] = str(pathlib.Path(checkpoint).absolute())


    config['num_classes'] = tfRecordsConfig['num_classes']


    #pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
    pipeline_config_path = os.path.join(modelOutput, "custom_pipeline.config")
    # Modify Pipeline file with the configuration data
    pipeline_config.setConfig(os.path.join(modelDir, "pipeline.config"), config, pipeline_config_path)

    # TF2
    tf.keras.backend.clear_session()
    tf.config.set_soft_device_placement(True)

    if num_workers > 1:
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.compat.v2.distribute.MirroredStrategy()

    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=pipeline_config_path,
            model_dir=modelOutput,
            train_steps=steps)


def eval(modelDir, dataDir, tfRecordsConfig=None, model="ssd_mobilenet_v2_coco_2018_03_29", steps=1000, eval_callback=None):
    if modelDir == None:
        modelDir = os.path.join("Traindata", "model", model)

    if not os.path.exists(modelDir):
        assert('Model not found!')

    # Create Pipeline Config
    config = {}

    # Load TFRecords
    if tfRecordsConfig is None:
        search = os.listdir(dataDir)
        for f in search:
            if f.startswith('Train.record'):
                config["train_input"] = str(pathlib.Path(os.path.join(dataDir, f)).absolute())
            if f.startswith('Eval.record'):
                config["eval_input"] = str(pathlib.Path(os.path.join(dataDir, f)).absolute())
                config["num_eval"] = 1
    else:
        if "train_input" in tfRecordsConfig:
            config["train_input"] = tfRecordsConfig["train_input"]

        if "eval_input" in tfRecordsConfig:
            config["eval_input"] = tfRecordsConfig["eval_input"]
            config["num_eval"] = tfRecordsConfig["num_eval"]
    config["label_map"] = str(pathlib.Path(os.path.join(dataDir, "labelmap.pbtxt")).absolute())

    # Check if there are checkpoints from older runs
    checkpointDir = os.path.join(modelDir, "checkpoint")
    finetune_checkpoint = ""
    for f in os.listdir(checkpointDir):
        if f.endswith(".index"):
            finetune_checkpoint = f[:-6]
    config["checkpoint"] = str(pathlib.Path(os.path.join(checkpointDir, finetune_checkpoint)).absolute())


    config['num_classes'] = tfRecordsConfig['num_classes']

    for f in os.listdir(os.path.join(modelDir)):
        if f.endswith(".config"):
            pipeline_config_path = os.path.join(modelDir, f)

    # Update Pipeline file with the configuration data
    pipeline_config.setConfig(pipeline_config_path, config, pipeline_config_path)


    # TF2
    tf.keras.backend.clear_session()
    tf.config.set_soft_device_placement(True)

    model_lib_v2.eval_continuously(
        pipeline_config_path= pipeline_config_path,
        model_dir= modelDir,
        train_steps= steps,
        sample_1_of_n_eval_examples= 1,
        sample_1_of_n_eval_on_train_examples=(1),
        checkpoint_dir=checkpointDir,
        wait_interval=1, timeout=1,
        callback=eval_callback)





