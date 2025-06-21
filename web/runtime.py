import os
import sys
import threading
import time
from PIL import Image
import torch

# Add project root to Python path to allow imports from TIC
_RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_RUNTIME_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from TIC.utils.serve import load_model
from TIC.utils.preprocess import get_transforms, get_class_to_idx
from TIC.utils.parameter import get_image_size

try:
    import pynvml
except ImportError:
    print("pynvml not found, GPU memory check will be skipped. To enable it, run: pip install pynvml")
    pynvml = None

MODEL_TYPE = 'vit-large'
WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, 'checkpoint/ViT_large_finetune_on_filtered_data_production_epoch5.pth')
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data/data_filtered_vit_base')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INACTIVITY_TIMEOUT = 5 * 60  # 5 minutes
GPU_MEMORY_THRESHOLD = 0.5

class ModelDaemon:
    def __init__(self):
        self.model = None
        self.transforms = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.timer = None
        self.lock = threading.Lock()

    def start(self):
        # This method should be called within a lock
        if self.model is None:
            global DEVICE
            if not check_cuda_available():
                DEVICE = 'cpu'
                print("CUDA is not available or occupied by another task. Using CPU instead.")
            else:
                DEVICE = 'cuda'
            
            print("Starting model daemon...")
            if not os.path.exists(WEIGHTS_PATH):
                raise FileNotFoundError(f"Checkpoint file not found at {WEIGHTS_PATH}. Please make sure the model checkpoint is available.")
            if not os.path.exists(DATA_DIR):
                raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. Please make sure the data directory is available.")

            print("Loading class mapping...")
            self.class_to_idx = get_class_to_idx(data_dir=DATA_DIR)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            num_classes = len(self.class_to_idx)

            print(f"Loading model '{MODEL_TYPE}'...")
            self.model = load_model(MODEL_TYPE, num_classes, WEIGHTS_PATH, DEVICE)
            self.model.eval()

            print("Getting image transformations...")
            current_image_size = get_image_size(MODEL_TYPE)
            self.transforms = get_transforms(data_dir=DATA_DIR, image_size=current_image_size)
            
            print("Model daemon started successfully.")
        self._reset_timer()

    def stop(self):
        with self.lock:
            if self.model is not None:
                print("Stopping model daemon due to inactivity...")
                del self.model
                self.model = None
                self.transforms = None
                self.class_to_idx = None
                self.idx_to_class = None
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                print("Model daemon stopped.")
            if self.timer:
                self.timer.cancel()
                self.timer = None

    def _reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(INACTIVITY_TIMEOUT, self.stop)
        self.timer.start()

    def predict(self, image: Image.Image):
        if self.model is None:
            raise Exception("Model is not loaded. This should not happen if using serve().")

        self._reset_timer()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transforms(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            class_name = self.idx_to_class[predicted.item()]
        return class_name, confidence.item()

daemon = ModelDaemon()

def is_daemon_running():
    """Checks if the model daemon is currently running."""
    return daemon.model is not None

def is_daemon_cuda():
    """Checks if the model daemon is running on a CUDA device."""
    return DEVICE == 'cuda' and is_daemon_running()

def check_gpu_memory():
    """
    Checks if the GPU memory usage is within the defined threshold.
    If the usage is over the threshold and there are running processes, it returns False.
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Check if used memory is more than the threshold
        if mem_info.used / mem_info.total > GPU_MEMORY_THRESHOLD:
            # Check for running processes
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if processes:
                print(f"GPU has running processes and memory usage is > {GPU_MEMORY_THRESHOLD*100:.0f}%. Used: {mem_info.used/1024**2:.0f}MB, Total: {mem_info.total/1024**2:.0f}MB")
                return False
    except Exception as e:
        print(f"Could not check GPU memory: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    return True

def check_cuda_available():
    """Checks if CUDA is available and not occupied by other task."""
    if not torch.cuda.is_available():
        return False
    return check_gpu_memory()
    
    

def serve(image: Image.Image):
    started = False
    with daemon.lock:
        if daemon.model is None:

            daemon.start()
            started = True

    prediction, confidence = daemon.predict(image)
    return prediction, confidence, started

def serve_batch(images: list[Image.Image]):
    started = False
    with daemon.lock:
        if daemon.model is None:
            daemon.start()
            started = True
        
    results = []
    for image in images:
        results.append(daemon.predict(image))
    return results, started