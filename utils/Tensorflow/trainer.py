import object_detection as od


dataset = os.path.join(os.getcwd(), "Dataset")

od.dataset_tools.create_coco_tf_record._create_tf_record_from_coco_annotations()