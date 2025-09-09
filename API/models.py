import datetime
import uuid
from typing import Tuple

import cv2
import numpy as np
from google.cloud import storage
from ultralytics import YOLO


class Models:
    """
    A class to manage and use two YOLO models for detection and segmentation.

    This class initializes an official YOLO model for motorcycle detection and a
    custom pre-trained model for image segmentation. It also handles interaction
    with Google Cloud Storage for uploading segmentation results.
    """

    def __init__(self, custom_model_path: str, official_model_name: str, bucket_name: str, bucket_save_path: str) -> None:
        """
        Initializes the Models class with model paths and GCS configuration.

        Args:
            custom_model_path (str): The local file path to the custom pre-trained YOLO model.
            official_model_name (str): The name of the official YOLO model to load (e.g., 'yolov8n.pt').
            bucket_name (str): The name of the Google Cloud Storage bucket.
            bucket_save_path (str): The destination path within the GCS bucket.
        """
        self.official_model: YOLO = YOLO(official_model_name)
        self.custom_model: YOLO = YOLO(custom_model_path)
        self.storage_client: storage.Client = storage.Client()
        self.bucket: storage.Bucket = self.storage_client.bucket(bucket_name)
        self.bucket_save_path: str = bucket_save_path
        self.motorcycle_class_index: int = 3

    def detect_motorcycle(self, image_path: str) -> bool:
        """
        Detects motorcycles in an image using the official YOLO model.

        Args:
            image_path (str): The path or URL to the input image.

        Returns:
            bool: True if a motorcycle is detected with a confidence of 0.85 or higher, False otherwise.
        """
        results = self.official_model.predict(
            source=image_path,
            classes=[self.motorcycle_class_index],
            conf=0.85,
            verbose=False
        )
        
        if results and len(results[0].boxes) > 0:
            return True
        return False

    def segment_and_upload(self, image_path: str) -> Tuple[str, int, str, float]:
        """
        Performs segmentation on an image, uploads the result to GCS, and returns a signed URL.

        This method uses the custom pre-trained model to segment an image. It then saves
        the resulting image with segmentation masks, uploads it to the configured GCS bucket,
        and generates a signed URL for access.

        Args:
            image_path (str): The path or URL to the input image.

        Returns:
            Tuple[str, int, float]: A tuple containing the signed URL of the uploaded image,
                                    the detected class ID, and the confidence score of the detection.
        
        Raises:
            ValueError: If no objects are detected for segmentation in the image.
        """
        segmentation_results = self.custom_model.predict(source=image_path, verbose=False, max_det=1, conf=0.5)

        if not segmentation_results or len(segmentation_results[0].boxes) == 0:
            raise ValueError("No objects were detected for segmentation.")

        most_confident_result = segmentation_results[0]
        detected_class: int = int(most_confident_result.boxes.cls[0].item())
        detected_class_name = most_confident_result.names[detected_class]
        confidence_score: float = float(most_confident_result.boxes.conf[0].item())

        segmented_image_array: np.ndarray = most_confident_result.plot()

        timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id: str = str(uuid.uuid4())
        file_name: str = f"{unique_id}_{timestamp}.jpg"
        try:
            local_temp_path: str = f"/tmp/{file_name}"
            cv2.imwrite(local_temp_path, segmented_image_array)

            blob_path: str = f"{self.bucket_save_path.strip('/')}/{file_name}"
            blob: storage.Blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(local_temp_path)

            signed_url: str = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(hours=8),
                method="GET",
            )
        except Exception as e:
            print(e)
            raise ValueError("Making File error")

        return signed_url, detected_class, detected_class_name, confidence_score