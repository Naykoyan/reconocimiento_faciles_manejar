import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from models import Models

app = FastAPI(
    title="YOLO Motorcycle Detection and Segmentation API",
    description="An API to first detect motorcycles in an image and then perform segmentation if a motorcycle is found.",
    version="1.0.0"
)

# It is recommended to load these values from environment variables for security and flexibility.
CUSTOM_MODEL_PATH = os.getenv("CUSTOM_MODEL_PATH", "/home/nipelu1005/Project/API/modelo/weights/best.pt")
OFFICIAL_MODEL_NAME = os.getenv("OFFICIAL_MODEL_NAME", "yolo11x.pt")
BUCKET_NAME = os.getenv("BUCKET_NAME", "vision-scooter-results")
BUCKET_SAVE_PATH = os.getenv("BUCKET_SAVE_PATH", "res")

model_handler = Models(
    custom_model_path=CUSTOM_MODEL_PATH,
    official_model_name=OFFICIAL_MODEL_NAME,
    bucket_name=BUCKET_NAME,
    bucket_save_path=BUCKET_SAVE_PATH
)

class ImageRequest(BaseModel):
    """
    Request body model for the image processing endpoint.
    """
    image_path: str = Field(..., example="https://path/to/your/image.jpg", description="URL or local path of the image to process.")

class SegmentationResponse(BaseModel):
    """
    Response body model for a successful segmentation.
    """
    detected_class: int = Field(..., example=0, description="The class ID detected by the segmentation model (0: Faciles, 1: Otras).")
    detected_class_name: str = Field(..., example="Faciles", description="The class ID detected by the segmentation model (0: Faciles, 1: Otras).")
    confidence: float = Field(..., example=0.95, description="The confidence score of the detection.")
    signed_url: str = Field(..., example="https://storage.googleapis.com/...", description="Signed URL to access the segmented image.")


@app.post("/process-image/", tags=["Image Processing"])
async def process_image(request: ImageRequest) -> dict:
    """
    Processes an image to detect a motorcycle and perform segmentation.

    This endpoint first checks for the presence of a motorcycle in the provided image.
    If a motorcycle is detected, it proceeds to perform segmentation using a custom model,
    uploads the result to Google Cloud Storage, and returns a signed URL.

    Args:
        request (ImageRequest): The request body containing the path to the image.

    Returns:
        dict: A dictionary with the result. If no motorcycle is found, it returns a message.
              If a motorcycle is found, it returns the segmentation details.
    
    Raises:
        HTTPException: Returns a 500 status code if any server-side error occurs
                       during the detection or segmentation process.
    """
    try:
        motorcycle_found = model_handler.detect_motorcycle(request.image_path)

        if not motorcycle_found:
            return {"message": "No se encontró ninguna motocicleta en la imagen."}

        signed_url, detected_class, detected_class_name, confidence = model_handler.segment_and_upload(request.image_path)
        
        response = SegmentationResponse(    
            detected_class=detected_class,
            detected_class_name=detected_class_name,
            confidence=confidence,
            signed_url=signed_url
        )
        return response.model_dump()

    except Exception as e:
        # In a production environment, you should log the error `e` for debugging purposes.
        raise HTTPException(status_code=500, detail=f"Hubo un error durante el procesamiento: {str(e)}")


"""
export GOOGLE_APPLICATION_CREDENTIALS="/home/nipelu1005/Project/API/llave.json"

"""