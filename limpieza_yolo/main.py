import json
import logging
import os
import threading
from queue import Queue
from time import sleep
from typing import Any, Dict, List

import cv2
import torch
from google.cloud import storage
from ultralytics import YOLO

# --- CONFIGURACIÓN GLOBAL ---

# Configuración de GCP
BUCKET_NAME: str = "vision-scooter-raw"
# ¡IMPORTANTE! Reemplaza esta ruta por la correcta en tu bucket.
IMAGE_LIST_FILE_PATH: str = "manifest_de_imagenes.txt"
PROCESSED_FILE_KEY: str = "procesados.jsonl"
REJECTED_FILE_KEY: str = "rechazados.txt"

# Configuración del Procesamiento
NUM_WORKER_THREADS: int = 1

# Configuración del Modelo YOLO
YOLO_MODEL_NAME: str = "yolo11x-seg.pt"
CONFIDENCE_THRESHOLD: float =0.8
# ID de clase para 'motorcycle' en el dataset COCO es 3
TARGET_CLASS_ID: int = 3

# Configuración del Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- DEFINICIÓN DE TIPOS PARA CLARIDAD ---
DetectionData = Dict[str, Any]
ImagePathQueue = Queue[str]
ProcessedQueue = Queue[DetectionData]
RejectedQueue = Queue[str]

# --- CLIENTE DE GOOGLE CLOUD STORAGE ---
# Se asume que la autenticación está configurada en el entorno.
# Ejemplo: `gcloud auth application-default login`
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
except Exception as e:
    logging.critical(f"Error al inicializar el cliente de GCP: {e}")
    exit(1)


def gcs_writer_thread(
    queue: Queue,
    file_key: str,
    stop_event: threading.Event
) -> None:
    """
    Hilo genérico para escribir líneas en un archivo de texto en GCS.
    Espera elementos de una cola y los escribe en un blob.
    """
    logging.info(f"Hilo de escritura para '{file_key}' iniciado.")
    content_buffer: List[str] = []
    blob = bucket.blob(file_key)
    # Limpia el archivo al inicio
    blob.upload_from_string("", content_type="text/plain")

    while not (stop_event.is_set() and queue.empty()):
        try:
            item = queue.get(timeout=1)
            line = json.dumps(item) if isinstance(item, dict) else str(item)
            content_buffer.append(line + "\n")

            # Escribe en el bucket en lotes para reducir las llamadas a la API
            if len(content_buffer) >= 10:
                existing_content = blob.download_as_text() if blob.exists() else ""
                blob.upload_from_string(existing_content + "".join(content_buffer))
                content_buffer.clear()

            queue.task_done()
        except Exception:
            # La cola está vacía, continúa esperando
            continue

    # Escribe cualquier elemento restante en el buffer
    if content_buffer:
        existing_content = blob.download_as_text() if blob.exists() else ""
        blob.upload_from_string(existing_content + "".join(content_buffer))

    logging.info(f"Hilo de escritura para '{file_key}' finalizado.")


def image_processor_thread(
    image_queue: ImagePathQueue,
    processed_queue: ProcessedQueue,
    rejected_queue: RejectedQueue,
    model: YOLO
) -> None:
    """
    Hilo trabajador que descarga, procesa una imagen con YOLO y encola el resultado.
    """
    while not image_queue.empty():
        try:
            image_gcs_path:str = image_queue.get_nowait()
        except Exception:
            break  # La cola está vacía

        logging.info(f"Procesando: {image_gcs_path}")
        local_path = os.path.join("/tmp", os.path.basename(image_gcs_path))

        try:
            # 1. Descargar la imagen
            bucket.blob(image_gcs_path.replace("gs://vision-scooter-raw/","")).download_to_filename(local_path)

            # 2. Leer y convertir la imagen
            img = cv2.imread(local_path)
            if img is None:
                raise ValueError("No se pudo leer el archivo de imagen.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 3. Procesar con YOLO
            results = model.predict(
                source=img_rgb,
                conf=CONFIDENCE_THRESHOLD,
                classes=[TARGET_CLASS_ID],
                verbose=False
            )
            result = results[0]  # Primer resultado del lote

            # 4. Si no hay detecciones, enviar a rechazados
            if result.masks is None or len(result.masks) == 0:
                logging.warning(f"Sin detecciones para {image_gcs_path}. Rechazado.")
                rejected_queue.put(image_gcs_path)
                continue

            # 5. Si hay detecciones, estructurar los datos
            detections: List[Dict[str, List]] = []
            for box, mask in zip(result.boxes, result.masks):
                detections.append({
                    "box_xyxy": box.xyxy.tolist()[0],
                    "box_xywh": box.xywh.tolist()[0],
                    "box_xyxyn": box.xyxyn.tolist()[0],
                    "box_xywhn": box.xywhn.tolist()[0],
                    "mask_xy": mask.xy[0].tolist(),
                    "mask_xyn": mask.xyn[0].tolist(),
                })

            # 6. Crear el diccionario final
            output_data: DetectionData = {
                "nombre": os.path.basename(image_gcs_path),
                "path": image_gcs_path,
                "cantidad_detecciones": len(detections),
                "detecciones": detections,
            }

            # 7. Enviar a la cola de procesados
            processed_queue.put(output_data)
            logging.info(f"Procesado con éxito: {image_gcs_path} ({len(detections)} detecciones)")

        except Exception as e:
            logging.error(f"Error procesando {image_gcs_path}: {e}")
            rejected_queue.put(image_gcs_path)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)
            image_queue.task_done()


def main() -> None:
    """
    Función principal que orquesta la lectura, procesamiento y escritura.
    """
    # 1. Abrir archivo de texto y encolar rutas
    logging.info("Iniciando el proceso de detección y segmentación.")
    image_queue: ImagePathQueue = Queue()
    try:
        blob = bucket.blob(IMAGE_LIST_FILE_PATH)
        image_list_content = blob.download_as_text()
        image_paths = [path.strip() for path in image_list_content.splitlines() if path.strip()]
        if not image_paths:
            logging.critical("El archivo de lista de imágenes está vacío o no existe.")
            return
        for path in image_paths:
            image_queue.put(path)
        logging.info(f"Se encolaron {len(image_paths)} imágenes para procesar.")
    except Exception as e:
        logging.critical(f"No se pudo leer la lista de imágenes '{IMAGE_LIST_FILE_PATH}': {e}")
        return

    # 2. Crear colas de resultados
    processed_queue: ProcessedQueue = Queue()
    rejected_queue: RejectedQueue = Queue()
    stop_event = threading.Event()

    # Cargar el modelo YOLO una sola vez
    logging.info(f"Cargando modelo YOLO: {YOLO_MODEL_NAME}")
    try:
        model = YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        logging.critical(f"No se pudo cargar el modelo YOLO. Error: {e}")
        return
    logging.info("Modelo YOLO cargado exitosamente.")

    # 3. Iniciar hilos de escritura
    processed_writer = threading.Thread(
        target=gcs_writer_thread,
        args=(processed_queue, PROCESSED_FILE_KEY, stop_event),
        name="ProcessedWriter"
    )
    rejected_writer = threading.Thread(
        target=gcs_writer_thread,
        args=(rejected_queue, REJECTED_FILE_KEY, stop_event),
        name="RejectedWriter"
    )
    processed_writer.start()
    rejected_writer.start()

    # 5. Iniciar hilos de procesamiento
    worker_threads: List[threading.Thread] = []
    logging.info(f"Iniciando {NUM_WORKER_THREADS} hilos de procesamiento...")
    for i in range(NUM_WORKER_THREADS):
        worker = threading.Thread(
            target=image_processor_thread,
            args=(image_queue, processed_queue, rejected_queue, model),
            name=f"Worker-{i+1}"
        )
        worker_threads.append(worker)
        worker.start()
        sleep(1)

    # 6. Esperar a que todos los hilos concluyan
    # Esperar a que se procesen todas las imágenes
    image_queue.join()
    logging.info("Todos los hilos de procesamiento han finalizado.")

    # Notificar a los hilos de escritura que pueden terminar
    stop_event.set()

    # Esperar a que las colas de escritura se vacíen
    processed_queue.join()
    rejected_queue.join()

    # Esperar a que los hilos de escritura terminen su ejecución
    processed_writer.join()
    rejected_writer.join()

    logging.info("Proceso completado. Archivos de salida generados en el bucket.")


if __name__ == "__main__":
    main()