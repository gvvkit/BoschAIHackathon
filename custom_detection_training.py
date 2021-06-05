# import DetectionModelTrainer from ImageAI
from imageai.Detection.Custom import DetectionModelTrainer

# Create instance of class DetectionModelTrainer and set this to trainer
trainer = DetectionModelTrainer()
# Set model type to YOLOv3
trainer.setModelTypeAsYOLOv3()
# Set path of custom dataset
trainer.setDataDirectory(data_directory="robotcaptures")
# Set the parameters
# Object name array add -- set all the object names you have annotated in LabelIMg, PascalVOC, data annotation.
# Batch size for training the image, large the batch size better is the accuracy, for memory sake I have kept them as 4
# num_experiments is the number of times we want the training code to iterate on custom dataset
# This training is conducted from a pretrained yolov3 model
trainer.setTrainConfig(object_names_array=["chair", "socks", "cable", "sofa", "teatable"], batch_size=2, num_experiments=6,
                       train_from_pretrained_model="detection_model-ex-005--loss-0023.575.h5")
# Strat the training
trainer.trainModel()

# Once the training is going on you will see the below as output
# Epoch 1/100
# 1/480 [..............................] - ETA: 1:37:23 - loss: 180.2602 - yolo_layer_loss: 26.1518 - yolo_layer_1_loss: 55.3661 - yolo_layer_2_loss
# 2021-06-05 13:23:41.285843: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.
# 2/480 [..............................] - ETA: 1:22:16 - loss: 178.9675 - yolo_layer_loss: 25.6380 - yolo_layer_1_loss: 55.0703 - yolo_layer_2_loss
# 2021-06-05 13:23:51.727791: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
# 372/480 [======================>.......] - ETA: 17:04 - loss: 68.9182 - yolo_layer_loss: 8.5295 - yolo_layer_1_loss: 17.5749 - yolo_layer_2_loss
# 374/480 [======================>.......] - ETA: 16:47 - loss: 68.7769 - yolo_layer_loss: 8.5112 - yolo_layer_1_loss: 17.5331 - yolo_layer_2_loss

# Output will generate detection_config.json file in the folder robotcapture/json, this file will be used during detection of objects in images
# ImageAI will created robotcapture/models folder which is where all the generated models will be saved
