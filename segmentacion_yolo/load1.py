import json
import multiprocessing
import os
import random
import time
from multiprocessing import JoinableQueue, Process
from typing import Dict, Any, List, Optional, TypedDict

import cv2
import numpy as np
from google.cloud import storage
from ultralytics import YOLO

# --- Tipos de Datos Estructurados ---

class Task(TypedDict):
    """Define la estructura de una tarea de procesamiento."""
    name: str
    path: str
    class_id: int
    local_path: str

class Result(TypedDict):
    """Define la estructura del resultado de la segmentación."""
    name: str
    class_id: int
    model_name: str
    local_path: str
    tmp_path: str
    xyn_results: List[np.ndarray]

# --- Lógica de Procesos de Segmentación ---

def segment_process(
    model_path: str,
    model_name: str,
    task_queue: JoinableQueue,
    result_queue: JoinableQueue,
) -> None:
    """
    Proceso que carga un modelo YOLO, procesa imágenes de una cola
    y envía los resultados a otra cola.
    """
    print(f"[{model_name}] Iniciando proceso de segmentación...")
    model = YOLO(model_path)
    CONFIDENCE_THRESHOLD: float = 0.8
    TARGET_CLASS: int = 3

    while True:
        task = task_queue.get()
        if task is None:
            print(f"[{model_name}] No hay más tareas. Terminando.")
            task_queue.task_done()
            break

        try:
            print(f"[{model_name}] Procesando: {task['name']}")
            img: np.ndarray = cv2.imread(task["local_path"])
            if img is None:
                print(f"[{model_name}] Error: No se pudo leer la imagen {task['local_path']}")
                task_queue.task_done()
                continue
            
            # Realiza la detección y segmentación
            results = model(img, conf=CONFIDENCE_THRESHOLD, classes=[TARGET_CLASS], verbose=False)

            overlay: np.ndarray = np.zeros_like(img, dtype=np.uint8)
            xyn_results: List[np.ndarray] = []
            
            # Procesa cada detección
            for result in results:
                if result.masks is None:
                    continue
                
                for mask, box, mask_xyn in zip(result.masks.xy, result.boxes, result.masks.xyn):
                    # Dibuja la máscara en el overlay con un color aleatorio
                    color = [random.randint(0, 255) for _ in range(3)]
                    cv2.fillPoly(overlay, [np.int32(mask)], color)
                    
                    # Dibuja el contorno sobre la imagen original
                    cv2.polylines(img, [np.int32(mask)], isClosed=True, color=color, thickness=2)
                    
                    # Guarda las coordenadas normalizadas de la máscara
                    xyn_results.append(mask_xyn)

            # Combina la imagen original con el overlay
            blended_img: np.ndarray = cv2.addWeighted(img, 1, overlay, 0.5, 0)
            
            # Guarda la imagen procesada temporalmente
            tmp_filename: str = f"{os.path.splitext(task['name'])[0]}_{model_name}.jpg"
            tmp_path: str = os.path.join("/tmp", tmp_filename)
            cv2.imwrite(tmp_path, blended_img)

            # Prepara el objeto de resultado
            output_data: Result = {
                "name": task["name"],
                "class_id": task["class_id"],
                "model_name": model_name,
                "local_path": task["local_path"],
                "tmp_path": tmp_path,
                "xyn_results": xyn_results,
            }
            
            result_queue.put(output_data)

        except Exception as e:
            print(f"[{model_name}] Error procesando {task['name']}: {e}")
        finally:
            task_queue.task_done()

# --- Lógica del Proceso de Carga a Cloud Storage ---

def upload_process(result_queue: JoinableQueue) -> None:
    """
    Proceso que toma resultados de una cola y los sube a Google Cloud Storage.
    """
    print("[UPLOADER] Iniciando proceso de carga...")
    storage_client: storage.Client = storage.Client()
    bucket_name: str = "vision-scooter-raw" # Reemplaza con el nombre de tu bucket
    bucket = storage_client.bucket(bucket_name)

    while True:
        result = result_queue.get()
        if result is None:
            print("[UPLOADER] No hay más resultados. Terminando.")
            result_queue.task_done()
            break
        
        try:
            model_name: str = result["model_name"]
            file_name: str = result["name"]
            base_name: str = os.path.splitext(file_name)[0]
            target_class:str = result["class_id"]
            
            # 1. Subir imagen original
            blob_original = bucket.blob(f"finales/{model_name}/images/{file_name}")
            blob_original.upload_from_filename(result["local_path"])
            
            # 2. Subir imagen con overlay (tags)
            blob_tagged = bucket.blob(f"finales/{model_name}/tags/{file_name}")
            blob_tagged.upload_from_filename(result["tmp_path"])
            
            # 3. Crear y subir archivo de etiquetas (labels)
            tmp_label_path = f"/tmp/{base_name}_{model_name}.txt"
            with open(tmp_label_path, "w") as f:
                for xyn_mask in result["xyn_results"]:
                    # La clase es la detectada (TARGET_CLASS), no la de la tarea
                    print(xyn_mask[0])
                    line_data = [str(target_class)] + [f"{coord:.6f}" for point in xyn_mask for coord in point]
                    f.write(" ".join(line_data) + "\n")

            blob_labels = bucket.blob(f"finales/{model_name}/labels/{base_name}.txt")
            blob_labels.upload_from_filename(tmp_label_path)
            
            print(f"[UPLOADER] Subido con éxito: {file_name} ({model_name})")

            # 5. Limpiar archivos temporales
            os.remove(result["tmp_path"])
            os.remove(tmp_label_path)

        except Exception as e:
            print(f"[UPLOADER] Error subiendo {result['name']}: {e}")
        finally:
            result_queue.task_done()

# --- Lógica Principal ---

def main() -> None:
    """
    Función principal que orquesta la creación de colas, procesos y la distribución de tareas.
    """
    start_time: float = time.time()
    
    # Rutas y configuración
    JSONL_PATH: str = "/home/nipelu1005/media/procesados.jsonl"
    YOLO8_MODEL_PATH: str = "models/yolov8x-seg.pt"
    YOLO11_MODEL_PATH: str = "models/yolo11x-seg.pt" # Asegúrate que este modelo exista
    
    # 1. Crear las colas de comunicación
    yolo8_queue: JoinableQueue[Optional[Task]] = JoinableQueue()
    yolo11_queue: JoinableQueue[Optional[Task]] = JoinableQueue()
    upload_queue: JoinableQueue[Optional[Result]] = JoinableQueue()
    
    # 2. Leer y distribuir tareas desde el archivo JSONL
    tasks_count: int = 0
    try:
        with open(JSONL_PATH, "r") as f:
            for line in f:
                data: Dict[str, Any] = json.loads(line)
                class_id = data.get("clase")
                if class_id in [0, 1]:
                    task: Task = {
                        "name": data["nombre"],
                        "path": data["path"],
                        "class_id": class_id,
                        "local_path": data["path"].replace("gs://vision-scooter-raw/", "/home/nipelu1005/media/"),
                    }
                    yolo8_queue.put(task)
                    yolo11_queue.put(task)
                    tasks_count += 1
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {JSONL_PATH}")
        return

    print(f"Se han cargado {tasks_count} tareas en cada cola de segmentación.")

    if tasks_count == 0:
        print("No hay tareas para procesar. Finalizando.")
        return

    # 3. Crear e iniciar los procesos
    processes: List[Process] = []

    # Procesos de segmentación
    p_yolo8 = Process(target=segment_process, args=(YOLO8_MODEL_PATH, "yolov8", yolo8_queue, upload_queue))
    p_yolo11 = Process(target=segment_process, args=(YOLO11_MODEL_PATH, "yolov11", yolo11_queue, upload_queue))
    
    # Proceso de carga
    p_uploader = Process(target=upload_process, args=(upload_queue,))
    
    processes.extend([p_yolo8, p_yolo11, p_uploader])

    for p in processes:
        p.start()
        
    # 4. Esperar a que las colas de tareas se vacíen
    print("Hilo principal esperando a que los procesos de segmentación terminen...")
    yolo8_queue.join()
    yolo11_queue.join()
    print("Todas las tareas de segmentación han sido procesadas.")
    
    # 5. Señalizar a los procesos de segmentación que terminen
    yolo8_queue.put(None)
    yolo11_queue.put(None)
    
    # 6. Esperar a que la cola de carga se vacíe
    print("Hilo principal esperando a que el proceso de carga termine...")
    upload_queue.join()
    print("Todos los resultados han sido subidos.")
    
    # 7. Señalizar al proceso de carga que termine
    upload_queue.put(None)

    # 8. Esperar a que todos los procesos finalicen completamente
    for p in processes:
        p.join()
        
    end_time: float = time.time()
    print(f"\nProceso completado en {end_time - start_time:.2f} segundos.")


if __name__ == "__main__":
    # Necesario para crear procesos de forma segura en algunos sistemas operativos
    multiprocessing.set_start_method("spawn")
    main()