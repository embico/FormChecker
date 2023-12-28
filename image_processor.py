import numpy as np
from matplotlib import pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import pandas as pd
import tensorflow as tf


print("start of image_processor")

class Processor:

    def __init__(self, fileName) -> None:
        self.capture = cv2.VideoCapture(fileName)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def print(self):
        print("hello, image processor here")

    def showPoints(self, lm, lmPose, frame):
        h, w = frame.shape[:2]

        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, (230, 156, 113), 4)

        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, (230, 156, 113), 4)

        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
        cv2.circle(frame, (l_elbow_x, l_elbow_y), 7, (230, 156, 113), 4)

        r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
        r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
        cv2.circle(frame, (r_elbow_x, r_elbow_y), 7, (230, 156, 113), 4)

        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
        cv2.circle(frame, (l_wrist_x, l_wrist_y), 7, (230, 156, 113), 4)

        r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
        r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
        cv2.circle(frame, (r_wrist_x, r_wrist_y), 7, (230, 156, 113), 4)

        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
        cv2.circle(frame, (r_hip_x, r_hip_y), 7, (230, 156, 113), 4)

        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        cv2.circle(frame, (l_hip_x, l_hip_y), 7, (230, 156, 113), 4)

        r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
        r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
        cv2.circle(frame, (r_knee_x, r_knee_y), 7, (230, 156, 113), 4)

        l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
        cv2.circle(frame, (l_knee_x, l_knee_y), 7, (230, 156, 113), 4)

        r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)
        cv2.circle(frame, (r_ankle_x, r_ankle_y), 7, (230, 156, 113), 4)

        l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
        cv2.circle(frame, (l_ankle_x, l_ankle_y), 7, (230, 156, 113), 4)

        r_toe_x = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x * w)
        r_toe_y = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].y * h)
        cv2.circle(frame, (r_toe_x, r_toe_y), 7, (230, 156, 113), 4)

        l_toe_x = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w)
        l_toe_y = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].y * h)
        cv2.circle(frame, (l_toe_x, l_toe_y), 7, (230, 156, 113), 4)

        cv2.line(frame, (l_elbow_x, l_elbow_y), (l_wrist_x, l_wrist_y), (230, 216, 173), 4)
        cv2.line(frame, (r_elbow_x, r_elbow_y), (r_wrist_x, r_wrist_y), (230, 216, 173), 4)

        cv2.line(frame, (l_elbow_x, l_elbow_y), (l_shldr_x, l_shldr_y), (230, 216, 173), 4)
        cv2.line(frame, (r_elbow_x, r_elbow_y), (r_shldr_x, r_shldr_y), (230, 216, 173), 4)

        cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (230, 216, 173), 4)
        cv2.line(frame, (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y), (230, 216, 173), 4)

        cv2.line(frame, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), (230, 216, 173), 4)
        cv2.line(frame, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), (230, 216, 173), 4)
        
        cv2.line(frame, (l_ankle_x, l_ankle_y), (l_knee_x, l_knee_y), (230, 216, 173), 4)
        cv2.line(frame, (r_ankle_x, r_ankle_y), (r_knee_x, r_knee_y), (230, 216, 173), 4)

        cv2.line(frame, (l_ankle_x, l_ankle_y), (l_toe_x, l_toe_y), (230, 216, 173), 4)
        cv2.line(frame, (r_ankle_x, r_ankle_y), (r_toe_x, r_toe_y), (230, 216, 173), 4)
        
    def isFacingRight(self, lm, lmPose, frame):
        h, w = frame.shape[:2]
        r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)

        r_toe_x = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x * w)
        l_toe_x = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w)

        return r_ankle_x < r_toe_x and l_ankle_x < l_toe_x
        
    def isAtSquatDepth(self, lm, lmPose, frame):
        h, w = frame.shape[:2]
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

        return r_hip_y > r_knee_y and l_hip_y > l_knee_y

    def poseAnalysis(self, frame):

        keypoints = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm = keypoints.pose_landmarks
        lmPose = self.mp_pose.PoseLandmark
        
        if lm is None:
            print("person not found")
            return frame

        self.showPoints(lm, lmPose, frame)

        if self.isFacingRight(lm, lmPose, frame):
            self.writeAtPos("Facing: Right", 0, frame)
        else:
            self.writeAtPos("Facing: Left", 0, frame)

        if self.isAtSquatDepth(lm, lmPose, frame):
            self.writeAtPos("Hit Depth", 1, frame)
        else:
            self.writeAtPos("Did Not Hit Depth", 1, frame)
        
        return frame


    def MPFrame(self, frame):
        #model_path = 'efficientdet_lite0.tflite'
        model_path = 'detect.tflite'

        # Get the full absolute path to the model file
        absolute_model_path = os.path.abspath(model_path)

        # Check if the file exists
        if os.path.exists(absolute_model_path):
            # Try to open the file to see if there are any issues
            try:
                with open(absolute_model_path, 'rb') as file:
                    print(f"File '{absolute_model_path}' successfully opened.")
            except Exception as e:
                print(f"Error opening the file: {e}")
        else:
            print(f"File '{absolute_model_path}' does not exist.")

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        base_options = BaseOptions(model_asset_path=model_path)
        options = ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.1,
                                       running_mode=VisionRunningMode.IMAGE)
        
        
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detector = vision.ObjectDetector.create_from_options(options)
        detection_result = detector.detect(image)
        image_copy = np.copy(image.numpy_view())
        annotated_image = self.visualize(image_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return rgb_annotated_image
    
    def writeAtPos(self, text, pos, frame):
        cv2.putText(frame, text, (10, 70 * (pos + 1)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 50), 3, cv2.LINE_AA)

    def read(self):
        while True:
            i = 0
            foundFrame, frame = self.capture.read()
            if not foundFrame:
                    print("Null frame found")
                    break
            output = frame.copy()
            self.poseAnalysis(output)
            #output = self.MPFrame(frame)
            cv2.imshow('Video', output)

            if cv2.waitKey(20) & 0xFF==ord('d'):
                break

        self.capture.release()
        cv2.destroyAllWindows()
    def load_model(self, model_path):
        # Load your TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def process_frame(interpreter, frame, detection_threshold=0.5):
        results = get_bounding_boxes(interpreter, detection_threshold)
        return results

    def tfRead(self):
        model_path = 'detect.tflite'
        interpreter = self.load_model(model_path)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Perform analysis on each frame
            results = process_frame(interpreter, frame)

            # Visualize the results on the frame
            # ...
            # Display the frame
            cv2.imshow('Video Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def visualize(
            self,
            image,
            detection_result
        ) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.
        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to be visualize.
        Returns:
            Image with bounding boxes.
        """
        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (255, 0, 0)  # red
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                            MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return image
