import json
import numpy as np
import os
import logging

def jsonAnnotaions(uuid, thermalData, faces, outputPath):
    """
    Sript to write Annotation Files
    Input: 
    uuid: uuid for this annoation, same as image and thermalimage uuid
    thermalData: array with raw data from thermalcamera
    faces: output from detection models (Keys: type, xmin, xmax, ymin, ymax)
    outputPath: folder where the annoations are stored
    """
    logging.info("Creat Annotaionset")
    # Create Annotationset
    annotation = {}
    annotation["_id"] = uuid
    annotation["image"] = uuid + ".jpg"
    annotation["thermalImage"] = uuid + ".jpg"

    # Prepare Temperature Data
    data = np.asarray(thermalData)
    data = data.reshape(60, 80) #original output size from ThermalCamera

    # Create object list
    objects = []
    for (x, y, w, h) in faces['detection_boxes']:
        face = {}
        face["type"] = "Face"

        # Map Bounding Box
        bbox = {}
        bbox["xmin"] = x
        bbox["ymin"] = y
        bbox["xmax"] = w
        bbox["ymax"] = h
        face["bbox"] = bbox

        # Add Metadata
        meta = {}
        rawdata = data[int(y*60):int(h*60), int(x*80):int(w*80)]
        meta["rawData"] = rawdata.tolist()
        meta["median"] = int(np.median(meta["rawData"]))
        meta["max"] = int(np.amax(meta["rawData"]))
        meta["min"] = int(np.amin(meta["rawData"]))
        maxspot = np.where(rawdata == meta["max"])
        meta["maxspot"] = [int(maxspot[0]), int(maxspot[1])]
        meta["maxspotABS"] = [int(y*60+maxspot[0]), int(x*80+maxspot[1])]
        meta["maxspotABSNorm"] = [int((y*60+maxspot[0])/60), int((x*80+maxspot[1])/80)]
        meta["maxavg5"] = int(np.median(data[int(max(0, meta["maxspot"][0]-2)):int(min(60, meta["maxspot"][0]+2)), int(max(0, meta["maxspot"][1]-2)):int(min(80, meta["maxspot"][1]+2))])) # Average Temperature around the maxSpot with padding 10
        face["meta"] = meta
        objects.append(face)
    annotation["objects"] = objects

    with open(os.path.join(outputPath, uuid +".json"), 'w') as outfile:
        json.dump(annotation, outfile)
        
        

def previewAnnotaion(thermalData, faces, outputPath):
    """
    Sript to write Annotation Files
    Input: 
    thermalData: array with raw data from thermalcamera
    faces: output from all detection models (normal, thermal) (Keys: type, xmin, xmax, ymin, ymax)
    outputPath: folder where the annoations are stored
    """
    logging.info("Creat Annotaionset")
    # Create Annotationset
    annotation = {}

    # Prepare Temperature Data
    data = np.asarray(thermalData)
    data = data.reshape(60, 80) #original output size from ThermalCamera

    # Create object list
    
    objects = []
    # Faces
    for i, (x, y, w, h) in enumerate(faces[0]['detection_boxes']):
        if faces[1]['detection_scores'][i] < 0.2:
            continue
        face = {}
        face["type"] = "Face"

        # Map Bounding Box
        bbox = {}
        bbox["xmin"] = float(x)
        bbox["ymin"] = float(y)
        bbox["xmax"] = float(w)
        bbox["ymax"] = float(h)
        face["bbox"] = bbox
        objects.append(face)
        
    # Thermal Faces
    thermalObjects = []
    for i, (x, y, w, h) in enumerate(faces[1]['detection_boxes']):
        if faces[1]['detection_scores'][i] < 0.2:
            continue
        face = {}
        face["type"] = "Face"

        # Map Bounding Box
        bbox = {}
        bbox["xmin"] = float(x)
        bbox["ymin"] = float(y)
        bbox["xmax"] = float(w)
        bbox["ymax"] = float(h)
        face["bbox"] = bbox
        print(bbox)
        # Add Metadata
        meta = {}
        raw = data[int(y*60):int(h*60), int(x*80):int(w*80)]
        rawdata = raw.tolist()
        meta["median"] = int(np.median(rawdata))
        meta["max"] = int(np.amax(rawdata))
        meta["min"] = int(np.amin(rawdata))
        print(meta)
        maxspot = np.where(raw == meta["max"])
        print(maxspot[0][0])
        print(maxspot[1][0])
        meta["maxspot"] = [int(maxspot[0][0]), int(maxspot[1][0])]
        meta["maxspotABS"] = [int(y*60+meta["maxspot"][0]), int(x*80+meta["maxspot"][1])]
        meta["maxspotABSNorm"] = [int((y*60+meta["maxspot"][0])/60), int((x*80+meta["maxspot"][1])/80)]
        meta["maxavg5"] = int(np.median(data[int(max(0, meta["maxspot"][0]-2)):int(min(60, meta["maxspot"][0]+2)), int(max(0, meta["maxspot"][1]-2)):int(min(80, meta["maxspot"][1]+2))])) # Average Temperature around the maxSpot with padding 10
        face["meta"] = meta
        thermalObjects.append(face)
        
    annotation["objects"] = objects
    annotation["thermalObjects"] = thermalObjects
    print(annotation)
    with open(outputPath, 'w') as outfile:
        json.dump(annotation, outfile)