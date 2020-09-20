import utils.Tensorflow.tff as tff
import os
import time
import json
import numpy as np
import logging
#Log
logging.basicConfig(level=logging.DEBUG)
baseModel = os.path.join("models", "tf2_mobilenet")
base_checkpoint = os.path.join(baseModel, "checkpoint", "ckpt-0")
base_pipeline = os.path.join(baseModel, "pipeline.config")

model_id = "2601"
modelDir = os.path.join("checkpoints", "tf2_mobilenet_Demo2", model_id)
pipeline = os.path.join(modelDir, "pipeline.config")
checkpoints = []
checkpoint = os.path.join(modelDir, "checkpoint", "ckpt-5")
checkpoints.append(checkpoint)
chkps = []
aggregatedData = {}
t1 = time.time()

base = tff.readCheckpointValues((base_pipeline, base_checkpoint , 0))
t2 = time.time()
logging.debug(f'Load Ref: {t2-t1}')


check = tff.readCheckpointValues((pipeline, checkpoint , 1))
out = tff.delta(base, check)


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

with open('data.json', 'w') as f:
    json.dump(out, f, default=default)