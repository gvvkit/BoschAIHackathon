# Important point from training, lower the validation loss,  eg : loss 5.24453, more acurate the model will be
# This is to evaluate the models and pick the mAP (mean Average Precision) of all the models saved in hololens/models folder
# Higher the mAP, better the detection accuracy of the model
# import DetectionModelTrainer from ImageAI
from imageai.Detection.Custom import  DetectionModelTrainer

# Create instance of class DetectionModelTrainer and set this to trainer
trainer = DetectionModelTrainer()
# Set model type to YOLOv3
trainer.setModelTypeAsYOLOv3()
# Set path of custom dataset
trainer.setDataDirectory(data_directory="robotcaptures")
# Calling the evaluate model
# model path is the path where the models are stored
# json path the path where the detection_config.json is saved
# iou_threshold, is the desired minimum intersection  over union value of mAP computation, value ranges between 0.0 to 1.0
# object_threshold is our desired minimum class score for the mAP computatio, score range between 0.0 to 1.0
# nms_threshold, this is Non-maximum supression for the mAP compuation
trainer.evaluateModel(model_path="robotcaptures/models", json_path="robotcaptures/json/detection_config.json",
                      iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)

