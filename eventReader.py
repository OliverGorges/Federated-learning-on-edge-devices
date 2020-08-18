import tensorflow as tf
import os

evalDir = os.path.join( 'eval')
tags = ['DetectionBoxes_Percision/mAP', 'DetectionBoxes_Recall/AR@1', 'loss', 'Loss/classification_loss', 'Loss/localization_loss', 'Loss/regularization_loss', 'Loss/total_loss']
meta = {}    
evalFiles = os.listdir(evalDir)
print(evalFiles)
evalMeta = {}
for i, evalFile in enumerate(evalFiles):
    for e in tf.compat.v1.train.summary_iterator(os.path.join(evalDir, evalFile)):
        for v in e.summary.value:
            print('################\n')
            print(v.tag)
            print(v.simple_value)
            evalMeta[v.tag] = v.simple_value
    break
meta['eval'] = evalMeta

data = json.dumps(meta)
with open('meta.json', 'w') as outfile:
    json.dump(data, outfile)