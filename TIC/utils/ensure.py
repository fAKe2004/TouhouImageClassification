'''
Ensuring any hugging face model is downloaded
'''

import os.path

from huggingface_hub import snapshot_download

from TIC.utils.parameter import *

def ensure(model_name : str) -> str:
    target_path = os.path.join(CACHE_DIR, model_name)
    if not os.path.exists(target_path):
        snapshot_download(repo_id = model_name, local_dir = target_path)
    return target_path
