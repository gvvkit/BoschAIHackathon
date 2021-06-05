from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("robotcapture/models/detection_model-ex-005--loss-0023.575.h5")
detector.setJsonPath("robotcapture/json/detection_config.json")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="images/table.jpg", output_image_path="images/detecttableobj.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ",
          detection["box_points"])
