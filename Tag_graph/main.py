import threading
import queue
import json
import os
import random
import time
import logging
from pathlib import Path
from typing import Any

# --- Librerías de terceros ---
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import cv2 # Se añade OpenCV
import numpy as np # Necesario para OpenCV

# --- Configuración de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s',
    datefmt='%H:%M:%S'
)

# --- Constantes Globales ---
GCS_BUCKET_NAME: str = "vision-scooter-raw"
LOCAL_WORKSPACE: Path = Path("workspace")

# !!! IMPORTANTE: Modifica esta línea para apuntar a la carpeta que contiene tus imágenes !!!
IMAGE_DIRECTORY: Path = Path("/home/nipelu1005/media")

RESULTS_DIR: Path = Path("results")
MAX_REJECTIONS: int = 6
IMAGE_GRID_SIZE: tuple[int, int] = (3, 2)
BATCH_SIZE: int = IMAGE_GRID_SIZE[0] * IMAGE_GRID_SIZE[1]

# --- Definición de Tipos de Datos ---
ImageData = dict[str, Any]
ImageBatch = list[ImageData]

# --- Funciones de Soporte ---

def setup_environment():
    """Crea los directorios necesarios si no existen."""
    LOCAL_WORKSPACE.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    logging.info("Directorios de trabajo listos.")

def parse_local_data() -> list[ImageData]:
    """
    Lee el archivo .jsonl local y filtra los datos para 'dataset2'.
    """
    local_jsonl_path = IMAGE_DIRECTORY / "procesados.jsonl"
    image_data_list: list[ImageData] = []
    
    if not local_jsonl_path.exists():
        logging.error(f"FATAL: El archivo {local_jsonl_path} no se encontró.")
        return []

    logging.info(f"Leyendo metadatos desde: {local_jsonl_path}")
    with open(local_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "dataset" in data.get("path", "") and int(data.get("clase", 2)) == 1:
                    image_data: ImageData = {
                        "nombre": data.get("nombre"),
                        "path": data.get("path"),
                        "xyxy": [det.get("xyxy") for det in data.get("detecciones", []) if "xyxy" in det],
                        "rechazos": 0
                    }
                    image_data_list.append(image_data)
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Omitiendo línea mal formada en .jsonl: {e}")

    logging.info(f"Se procesaron {len(image_data_list)} registros de 'dataset1'.")
    return image_data_list

# --- Clase de la Interfaz Gráfica (GUI) ---

class ClassifierGUI:
    """
    Controla la ventana de Tkinter. Ya no es un hilo, sino que se ejecuta
    en el hilo principal.
    """
    def __init__(self, root: tk.Tk, image_list: list[ImageData], writer_queue: queue.Queue, rejection_queue: queue.Queue, lock: threading.Lock, stop_event: threading.Event):
        self.root = root
        self.image_list = image_list
        self.writer_queue = writer_queue
        self.rejection_queue = rejection_queue
        self.lock = lock
        self.stop_event = stop_event
        
        self.root.title("Clasificador de Imágenes")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.image_labels: list[tk.Label] = []
        self.current_batch: ImageBatch | None = None
        self.photo_images: list[ImageTk.PhotoImage] = []

        self._setup_ui()
        self.root.after(100, self._process_next_batch)

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        for i in range(BATCH_SIZE):
            row, col = divmod(i, IMAGE_GRID_SIZE[0])
            label = ttk.Label(main_frame, text=f"Imagen {i+1}", relief="solid", padding="5", anchor="center")
            label.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.image_labels.append(label)

        info_label = ttk.Label(
            self.root, 
            text="Clic Izquierdo: Aceptar lote | Clic Derecho: Rechazar lote completo",
            padding="5",
            font=("Arial", 10, "bold")
        )
        info_label.grid(row=1, column=0, pady=(5, 10))

    def _on_closing(self):
        logging.info("Ventana cerrada. Señalando a otros hilos para terminar.")
        self.stop_event.set()
        self.root.destroy()

    def _process_next_batch(self):
        if self.stop_event.is_set():
            return
            
        if not self._display_new_batch():
            logging.info("No hay más imágenes para mostrar. La GUI se cerrará en 3 segundos.")
            self.root.after(3000, self._on_closing)

    def _get_random_batch(self) -> ImageBatch | None:
        with self.lock:
            if not self.image_list:
                return None
            sample_size = min(len(self.image_list), BATCH_SIZE)
            return random.sample(self.image_list, sample_size) if sample_size > 0 else None

    def _display_new_batch(self) -> bool:
        self.current_batch = self._get_random_batch()

        if not self.current_batch:
            for label in self.image_labels:
                try:
                    label.config(image=None, text="FIN")
                except:
                    pass
                label.unbind("<Button-1>")
                label.unbind("<Button-3>")
            return False

        self.photo_images.clear()

        for i, img_data in enumerate(self.current_batch):
            label = self.image_labels[i]
            local_path_str = img_data["path"].replace(f"gs://{GCS_BUCKET_NAME}", str(IMAGE_DIRECTORY))
            
            try:
                # --- INICIO DE LA CORRECCIÓN CON OPENCV ---
                # 1. Leer la imagen con OpenCV
                img_bgr = cv2.imread(local_path_str)
                if img_bgr is None:
                    raise FileNotFoundError("OpenCV no pudo leer la imagen")
                
                original_height, original_width, _ = img_bgr.shape
                
                # 2. Redimensionar la imagen
                new_width, new_height = 400, 300
                img_resized = cv2.resize(img_bgr, (new_width, new_height))

                # 3. Calcular factores de escala
                x_scale = new_width / original_width
                y_scale = new_height / original_height

                # 4. Dibujar las cajas escaladas sobre la imagen redimensionada
                for box in img_data["xyxy"]:
                    pt1 = (int(box[0] * x_scale), int(box[1] * y_scale))
                    pt2 = (int(box[2] * x_scale), int(box[3] * y_scale))
                    cv2.rectangle(img_resized, pt1, pt2, (0, 0, 255), 2) # Color rojo en BGR

                # 5. Convertir la imagen de BGR (OpenCV) a RGB (Pillow/Tkinter)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # 6. Convertir el array de NumPy a una imagen de Pillow y luego a PhotoImage
                pil_img = Image.fromarray(img_rgb)
                photo = ImageTk.PhotoImage(image=pil_img)
                # --- FIN DE LA CORRECCIÓN ---
                
                self.photo_images.append(photo)
                
                label.config(image=photo)
                label.bind("<Button-1>", self._on_left_click)
                label.bind("<Button-3>", self._on_right_click)
            except FileNotFoundError:
                label.config(image=None, text=f"No encontrada:\n{Path(local_path_str).name}")
            except Exception as e:
                label.config(image=None, text=f"Error al cargar:\n{e}")
        
        for i in range(len(self.current_batch), BATCH_SIZE):
            try:
                self.image_labels[i].config(image=None, text="")
            except:
                pass
            self.image_labels[i].unbind("<Button-1>")
            self.image_labels[i].unbind("<Button-3>")
            
        return True

    def _on_left_click(self, event: Any):
        if self.current_batch:
            self.writer_queue.put(list(self.current_batch))
            self.root.after(100, self._process_next_batch)

    def _on_right_click(self, event: Any):
        if self.current_batch:
            for img_data in self.current_batch:
                self.rejection_queue.put(img_data["nombre"])
            self.root.after(100, self._process_next_batch)

# --- Hilos de Fondo ---

class RejectionManager(threading.Thread):
    """Gestiona los rechazos en segundo plano."""
    def __init__(self, image_list: list[ImageData], rejection_queue: queue.Queue, lock: threading.Lock, stop_event: threading.Event):
        super().__init__(name="Rejection-Manager-Thread")
        self.image_list = image_list
        self.rejection_queue = rejection_queue
        self.lock = lock
        self.stop_event = stop_event

    def run(self):
        logging.info("Hilo de gestión de rechazos iniciado.")
        while not self.stop_event.is_set():
            with self.lock:
                while not self.rejection_queue.empty():
                    try:
                        rejected_name: str = self.rejection_queue.get_nowait()
                        for img_data in self.image_list:
                            if img_data["nombre"] == rejected_name:
                                img_data["rechazos"] += 1
                                break
                        self.rejection_queue.task_done()
                    except queue.Empty:
                        break

                original_count = len(self.image_list)
                self.image_list[:] = [img for img in self.image_list if img["rechazos"] < MAX_REJECTIONS]
                if original_count - len(self.image_list) > 0:
                    logging.info(f"Limpieza: Se eliminaron {original_count - len(self.image_list)} imágenes.")
                
                if not self.image_list:
                    logging.info("Lista de imágenes vacía, gestor de rechazos terminará.")
                    break
            
            self.stop_event.wait(timeout=2.0)
            logging.info("Fin limpiando. Faltan:" + str(len(self.image_list)))
        logging.info("Hilo de gestión de rechazos finalizado.")

class ResultsWriter(threading.Thread):
    """Escribe los resultados en segundo plano."""
    def __init__(self, image_list: list[ImageData], writer_queue: queue.Queue, lock: threading.Lock, stop_event: threading.Event, results_file_path: Path):
        super().__init__(name="Writer-Thread")
        self.image_list = image_list
        self.writer_queue = writer_queue
        self.lock = lock
        self.stop_event = stop_event
        self.results_file_path = results_file_path

    def run(self):
        processed_names = self._load_existing_results()
        
        with self.lock:
            initial_count = len(self.image_list)
            self.image_list[:] = [img for img in self.image_list if img["nombre"] not in processed_names]
            logging.info(f"Escritor: {initial_count - len(self.image_list)} imágenes ya procesadas fueron omitidas.")

        while not self.stop_event.is_set():
            try:
                batch_to_write: ImageBatch = self.writer_queue.get(timeout=1)
                
                with open(self.results_file_path, 'a', encoding='utf-8') as f:
                    for img_data in batch_to_write:
                        if img_data["nombre"] not in processed_names:
                            f.write(json.dumps({"nombre": img_data["nombre"], "path": img_data["path"]}) + '\n')
                            processed_names.add(img_data["nombre"])
                
                batch_names_to_remove = {d["nombre"] for d in batch_to_write}
                with self.lock:
                    self.image_list[:] = [img for img in self.image_list if img["nombre"] not in batch_names_to_remove]
                
                self.writer_queue.task_done()
                logging.info("insertado otro batch")
            except queue.Empty:
                with self.lock:
                    if not self.image_list and self.writer_queue.empty():
                        break
        logging.info("Hilo de escritura finalizado.")

    def _load_existing_results(self) -> set[str]:
        if not self.results_file_path.exists():
            return set()
        
        with open(self.results_file_path, 'r', encoding='utf-8') as f:
            return {json.loads(line).get("nombre") for line in f if line.strip()}

# --- Función Principal (Hilo Principal) ---

def main():
    """Orquesta la inicialización y finalización de todos los hilos."""
    setup_environment()
    
    results_filename = input("Introduce el nombre del archivo de resultados (dejar en blanco para 'resultados_clasificados.jsonl'): ").strip()
    if not results_filename:
        results_filename = "resultados_clasificados.jsonl"
    if not results_filename.endswith(".jsonl"):
        results_filename += ".jsonl"
    
    results_file_path = RESULTS_DIR / results_filename
    logging.info(f"Los resultados se guardarán en: {results_file_path}")
    
    image_data_list = parse_local_data()
    if not image_data_list:
        logging.error("No se pudieron cargar los datos. El programa terminará.")
        return

    writer_queue = queue.Queue()
    rejection_queue = queue.Queue()
    data_lock = threading.Lock()
    stop_event = threading.Event()

    # Iniciar los hilos de fondo
    writer_thread = ResultsWriter(image_data_list, writer_queue, data_lock, stop_event, results_file_path)
    rejection_manager_thread = RejectionManager(image_data_list, rejection_queue, data_lock, stop_event)
    
    writer_thread.start()
    rejection_manager_thread.start()
    
    # Crear y ejecutar la GUI en el hilo principal
    root = tk.Tk()
    app = ClassifierGUI(root, image_data_list, writer_queue, rejection_queue, data_lock, stop_event)
    root.mainloop() # Esta línea bloquea hasta que la ventana se cierra

    # Una vez que la ventana se cierra, esperamos a que los hilos de fondo terminen.
    logging.info("Esperando a que los hilos de fondo terminen...")
    writer_thread.join()
    rejection_manager_thread.join()

    logging.info("Programa finalizado con éxito.")

if __name__ == "__main__":
    main()
