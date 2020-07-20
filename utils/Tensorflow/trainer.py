from object_detection.dataset_tools.create_coco_tf_record import _create_tf_record_from_coco_annotations
from utils.Dataset.cocoAnnotationConverter import  XmlConverter, JsonConverter
from utils.Dataset import pipeline_config
import os
from shutil import copy
import pathlib
import tensorflow as tf
import time
import logging 
from object_detection import model_hparams
from object_detection import model_lib
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
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
    """
    python object_detection/export_inference_graph.py \
        --input_type=${INPUT_TYPE} \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
        --output_directory=${EXPORT_DIR}
    """
    pipeline_config_path=""
    trained_checkpoint_prefix = ""
    for f in sorted(os.listdir(modelDir)):
        if f.endswith(".config"):
            pipeline_config_path = os.path.join(modelDir, f)
        if f.endswith(".meta"):
            trained_checkpoint_prefix = os.path.join(modelDir, f)[:-5]

    if not input_shape:
        input_shape = [None, 300, 300, 3]
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    exporter.export_inference_graph(
        "image_tensor", pipeline_config, trained_checkpoint_prefix,
        modelDir, input_shape=input_shape,
        write_inference_graph=True)


def trainer( modelOutput, dataDir, tfRecordsConfig=None, model="ssd_mobilenet_v2_coco_2018_03_29", steps=1000):
    #TF GPUConfig
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfConfig)

    
    modelDir = os.path.join("Traindata", "model", model)

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
    config["checkpoint"] = str(pathlib.Path(os.path.join(modelDir,  "model.ckpt")).absolute())

    
    config['num_classes'] = tfRecordsConfig['num_classes']

    # Modify Pipeline file with the configuration data
    pipeline_config.setConfig(os.path.join(modelDir, "pipeline.config"), config, os.path.join(modelOutput, "custom_pipeline.config"))



    if not os.path.exists(modelOutput):
        os.makedirs(modelOutput)

    # Prepare Training
    train_config = tf.estimator.RunConfig(modelOutput)
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=train_config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path=os.path.join(modelOutput, "custom_pipeline.config"),
        train_steps=steps,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=(5))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if False:
        if False:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if False:
            estimator.evaluate(input_fn,
                                steps=10,
                                checkpoint_path=tf.train.latest_checkpoint(modelDir))
        else:
            model_lib.continuous_eval(estimator, modelDir, input_fn,train_steps, name)
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

