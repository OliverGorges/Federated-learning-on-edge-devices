from pycocotools.coco import COCO
import os
import json
import numpy as np
import random
import cv2
import requests
import logging

dataDir = os.path.join("Traindata", "data")
annotations = os.path.join(dataDir, "cocoTrain29082020.json")
output = os.listdir(dataDir)
# Remove hidden files
for o in output:
    if o.startswith(".") or o.endswith('.json'):
        output.remove(o)

rnOfClients = 1

labels = {}
labels['bird'] = 16
labels['cat'] = 17
labels['dog'] = 18
labels['horse'] = 19
labels['sheep'] = 20
labels['cow'] = 21
labels['elephant'] = 22
labels['bear'] = 23
labels['zebra'] = 24
labels['giraffe'] = 25
labels['without_mask'] = 1
labels['with_mask'] = 2
labels['mask_weared_incorrect'] = 2

animals = ['without_mask', 'with_mask', 'mask_weared_incorrect']

with open(annotations) as f:
    cocoData = json.load(f)

print(cocoData['images'][0])
print(cocoData['annotations'][0]['category_id'])

#coco = COCO(annotations)

size = 500
id_to_image = {}
for img in cocoData['images']:
    id_to_image[img['id']] = img

def sanityCheck(trueAnno, result, image_meta):
    imageHeight = image_meta['height']
    imageWidth = image_meta['width']
    x = int(result["xmin"] * imageHeight)
    y = int(result["ymin"] * imageWidth)
    w = int(result["xmax"]*  imageHeight) - x
    h = int(result["ymax"] * imageWidth) - y
    delta = list(map(lambda a, b: abs(a-b), trueAnno, [x,y,w,h]))
    for d in delta:
        if d > 10:
            logging.warning(f'Delta: {delta}, Truth: {trueAnno}, Result: {[x,y,w,h]}')
            return

def normBox(bbox, image_id):
    image = id_to_image[image_id]
    height = image['height']
    width = image['width']

    x = bbox[0] / height
    y = bbox[1] / width
    w = bbox[2] /  height + y
    h = bbox[3] / width + x

    sanityCheck(bbox, {'xmin': x, 'ymin': y, 'xmax': w, 'ymax': h}, image)

    return {'xmin': x, 'ymin': y, 'xmax': w, 'ymax': h}

def normBox2(bbox, image_id):
    image = id_to_image[image_id]
    height = image['width']
    width = image['height']

    x = bbox[0] / height
    y = bbox[1] / width
    w = bbox[2] /  height + x
    h = bbox[3] / width + y

    sanityCheck(bbox, {'xmin': x, 'ymin': y, 'xmax': w, 'ymax': h}, image)

    return {'xmin': x, 'ymin': y, 'xmax': w, 'ymax': h}

def writeData(i, folder):
    if not os.path.exists(os.path.join(dataDir, folder, "Images")):
        os.makedirs(os.path.join(dataDir, folder, "Images"))
    if not os.path.exists(os.path.join(dataDir, folder, "Annotations")):
        os.makedirs(os.path.join(dataDir, folder, "Annotations"))
    imageFile = os.path.join(dataDir, folder, "Images", f"{i['image']['id']}.jpg")
    annoFile = os.path.join(dataDir, folder, "Annotations", f"{i['image']['id']}.json")
    name = i['image']['file_name'].split('\\')[-1:][0]
    if not os.path.exists(imageFile):
        im = cv2.imread(i['image']['file_name'])
        im = cv2.resize(im, (640,480))
        cv2.imwrite(imageFile, im)
        #img_data = requests.get(i['image']['coco_url']).content
        #with open(imageFile, 'wb') as handler:
            #handler.write(img_data)
    
    print(name)
    print(name[:-4])
    a = {'_id': name[:-4], 'image':  f"{i['image']['id']}.jpg", 'thermalImage':  f"{i['image']['id']}.jpg", 'height': 480,'width': 640,'objects': i['objects']}
    with open(annoFile, 'w') as outfile:
        json.dump(a, outfile)

def writeLabelMap(animal, selection):
    item = []
    item.append({
                "id": 1,
                "supercategory": "none",
                "name": "animal",
                "used": True
            })
    for s in  selection:
        item.append({
                "id": 1,
                "supercategory": "animal",
                "name": s,
                "used": False
            })
    with open(os.path.join(dataDir, animal, "label_map.json"), 'w') as outfile:
        json.dump({"item": item}, outfile)
info = {}


def convertSelection():
    gen_dist = (np.random.dirichlet((1, 1, 1),rnOfClients)* 100)
    animal_annotations = {}
    for animal in animals:
        animal_annotations[animal] = [{'image_id': a['image_id'], 'bbox': normBox(a['bbox'], a['image_id']), 'category_id': a['category_id'], '_id': a['id'], 'area': a['area'], 'iscrowd': a['iscrowd']} for a in cocoData['annotations'] if a['category_id'] == labels[animal]]


    for client_id in range(rnOfClients):
        client = f'client_{client_id}'
        if not os.path.exists(os.path.join(dataDir, client)):
            os.makedirs(os.path.join(dataDir, client))
        selection = ['without_mask', 'with_mask', 'mask_weared_incorrect']
        dist = gen_dist[client_id]
        print(selection)
        print(dist)
        info[client] = {"classes": selection, "distribution": dist}
        for e, animal in enumerate(selection):
            images_ids = {}
            for i in animal_annotations[animal]:
                i['type'] = animal
                if i['image_id'] in images_ids.keys():
                    images_ids[i['image_id']]['objects'].append(i)
                else:
                    meta = id_to_image[i['image_id']]
                    image = {'file_name': meta['file_name'], 'height': meta['height'], 'width': meta['width'], 'id': meta['id']}
                    images_ids[i['image_id']] = {'image': image, 'objects': [i]}

            dataset = [images_ids[k] for k in images_ids.keys()]

            train = dataset #random.sample(dataset, int(dist[e]))

            for i in train:
                writeData(i, client)

        writeLabelMap(client, selection)
        #for i in mixed:
        #    writeData(i, "mixed")
    with open(os.path.join(dataDir, "info.json"), 'w') as outfile:
            json.dump({"info": info}, outfile)


def convertDataset():
    images_ids = {}
    classes = {1:'with_mask', 2:'mask_weared_incorrect', 3:'without_mask'}
    for i in cocoData['images']:
        image = {'file_name': i['file_name'], 'height': i['height'], 'width': i['width'], 'id': i['id']}
        objects = []
        for a in cocoData['annotations']:
            if i['id'] == a['image_id']:
                objects.append({'image_id':i['id'], 'bbox': normBox2(a['bbox'], a['image_id']), 'type': classes[a['category_id']], 'id': a['id'], 'area': a['area'], 'iscrowd': a['iscrowd']})
        print({'image': image, 'objects': objects})
        images_ids[i['id']] = {'image': image, 'objects': objects}

    dataset = [images_ids[k] for k in images_ids.keys()]
    for i in dataset:
        writeData(i, "MaskJSON")

    #writeLabelMap("MaskJSON", classes)


if __name__ == "__main__":
    convertDataset()    