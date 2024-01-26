import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the pre-trained model
model_path = 'C:/Users/Amar/Desktop/Project/models/ssd_mobilenet_v1_coco_2018_01_28/saved_model'
detection_model = tf.saved_model.load(model_path)

# Load label map
label_map_path = 'C:/Users/Amar/Desktop/Project/models/ssd_mobilenet_v1_coco_2018_01_28/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Set up webcam
cap = cv2.VideoCapture(0)

while True:
    ret, image_np = cap.read()

    # Prepare the image for inference
    input_tensor = tf.convert_to_tensor([image_np], dtype=tf.uint8)
    detections = detection_model.signatures["serving_default"](input_tensor)

    # Create a copy of the image for visualization
    vis_image = image_np.copy()

    # Visualization of the results of a detection.
    for i in range(int(detections['num_detections'][0])):
        class_id = int(detections['detection_classes'][0][i])
        score = detections['detection_scores'][0][i]
        bbox = detections['detection_boxes'][0][i].numpy()

        if score > 0.5:  # Adjust the threshold as needed
            h, w, _ = vis_image.shape
            ymin, xmin, ymax, xmax = bbox
            ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)

            # Draw bounding box on the image
            cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display class label and score
            label = category_index[class_id]['name']
            label_str = f"{label}: {int(score * 100)}%"
            cv2.putText(vis_image, label_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', cv2.resize(vis_image, (800, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
