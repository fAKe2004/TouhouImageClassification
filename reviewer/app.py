import os
import uuid
import threading
import time
import glob
from flask import Flask, render_template, request, session, redirect, url_for
from datetime import datetime


app = Flask(__name__, static_folder='data')
app.secret_key = os.urandom(24)
DATA_DIR = 'data'

'''
Consider the following directory structure:
data/
├── label1/
│   ├── 1.jpg
│   ├── 2.pending
│   ├── 3.session_uid.timestamp.processing
│   ├── 4.jpg
│   ├── 5.invalid
│   └── 6.jpg
├── label2/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   └── 6.jpg

Here we represent each image as (label, id)

For each session we maintain two queue: "queue" and "history",
the "queue" is a list of processing file,
the "history" is a list of processed file.
'''

def get_status(label: str, img_id: str) -> dict:

    '''
    Args:
        label: the subdirectory name (e.g., 'label1')
        img_id: the image file name without extension (e.g., '3')
    Return:
        dict {
            'label',
            'id',
            'status',
            'session_uid',
            'timestamp',
            'path',
        }
    '''

    base_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Label directory {base_path} does not exist")
    
    # Look for files matching the img_id prefix
    for filename in os.listdir(base_path):
        if filename.startswith(f"{img_id}."):
            app.logger.info(f"Find {filename} for {(label, img_id)}")
            full_path = os.path.join(base_path, filename)
            return parse_file_info(label, full_path)

    app.logger.info("Find no matching file for label %s and img_id %s", label, img_id)

    # If no matching file is found
    return {
        'label': label,
        'id': img_id,
        'status': None,
        'session_uid': None,
        'timestamp': None,
        'path': None,
    }

def parse_file_info(label: str, filepath: str) -> dict:
    filename = os.path.basename(filepath)
    parts = filename.split('.')
    
    if parts[-1] == '.jpg':
        status = 'valid'
        session_uid = None
        timestamp = None
    elif parts[-1] == '.pending':
        status = 'pending'
        session_uid = None
        timestamp = None
    elif parts[-1] == '.invalid':
        status = 'invalid'
        session_uid = None
        timestamp = None
    else:
        # Handle .session_uid.timestamp.processing format
        if len(parts) == 4 and parts[-1] == 'processing':
            status = 'processing'
            session_uid = parts[1]
            timestamp = parts[2]
        else:
            status = 'unknown'
            session_uid = None
            timestamp = None

    return {
        'label': label,
        'id': parts[0],
        'status': status,
        'session_uid': session_uid,
        'timestamp': timestamp,
        'path': filepath,
    }


def processed2processing(label: str, img_id: str, valid: bool, session_uid: str) -> bool:
    base_path = os.path.join(DATA_DIR, label)
    pending_path = os.path.join(base_path, f"{img_id}.{'jpg' if valid else 'invalid'}")
    processing_path = os.path.join(base_path, f"{img_id}.{session_uid}.{int(time.time())}.processing")
    try:
        os.rename(pending_path, processing_path)
        return True
    except Exception as e:
        app.logger.error(f"Failed to rename {pending_path} to {processing_path}: {e}")
        return False


def mark_processed(label: str, img_id: str, img_path: str, valid: bool) -> bool:
    new_ext = 'jpg' if valid else 'invalid'
    try:
        app.logger.info("Try to rename %s to %s", img_path, os.path.join(DATA_DIR, label, f"{img_id}.{new_ext}"))
        os.rename(img_path, os.path.join(DATA_DIR, label, f"{img_id}.{new_ext}"))
        return True
    except Exception as e:
        app.logger.error(f"Failed to rename {img_path} to {os.path.join(DATA_DIR, label, f'{img_id}.{new_ext}')}: {e}")
        return False


def get_one_pending(session_uid):
    '''
    Acquire one pending image and get it processing...
    '''
    for label in os.listdir(DATA_DIR):
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.pending'):
                    base_name = file.rsplit('.', 1)[0]  # 去掉 .pending
                    pending_path = os.path.join(label_dir, file)

                    timestamp = int(time.time())
                    processing_file = f"{base_name}.{session_uid}.{timestamp}.processing"
                    processing_path = os.path.join(label_dir, processing_file)

                    try:
                        os.rename(pending_path, processing_path)
                        number = base_name.split('.')[0]  # 提取图片编号
                        return label, number, processing_path
                    except OSError:
                        continue
    return None


def release_processing_images():
    '''
    Release images that have been processing for more than 60 seconds
    '''
    while True:
        time.sleep(10)
        now = time.time()
        for filepath in glob.glob(os.path.join(DATA_DIR, '*', '*.processing')):
            dir_name = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            parts = filename.split('.')
            if len(parts) < 4 or parts[-1] != 'processing':
                continue
            try:
                timestamp = int(parts[-2])
                if now - timestamp > 60:
                    base_parts = parts[:-3]
                    pending_file = '.'.join(base_parts) + '.pending'
                    pending_path = os.path.join(dir_name, pending_file)
                    os.rename(filepath, pending_path)
            except ValueError:
                continue


threading.Thread(target=release_processing_images, daemon=True).start() # Launch the release thread 


def init_user_session():
    if 'queue' not in session:
        session['queue'] = []
    if 'history' not in session:
        session['history'] = []
    if 'session_uid' not in session:
        session['session_uid'] = str(uuid.uuid4())

    if not session['queue']:
        result = get_one_pending(session['session_uid'])
        if result:
            label, number, img_path = result
            session['queue'].append((label, number))

    session.modified = True


def get_reference(label):
    ref_path = os.path.join(DATA_DIR, label, '0.jpg')
    if os.path.exists(ref_path):
        return os.path.relpath(ref_path, DATA_DIR)
    return None

def count_images():
    valid = invalid = pending = 0
    for label in os.listdir(DATA_DIR):
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file == '0.jpg':
                    continue
                if file.endswith('.jpg'):
                    valid += 1
                elif file.endswith('.invalid'):
                    invalid += 1
                elif file.endswith('.pending'):
                    pending += 1
    return valid + invalid, valid + invalid + pending

def urlwrapper(path: str | None) -> str | None:
    if path is not None:
        return path.replace('\\', '/')


def reduce_len(x: list, l: int):
    return x[-l:]


@app.route('/', methods=['GET', 'POST'])
def index():
    init_user_session()

    session['queue'] = reduce_len(session['queue'], 10)
    session['history'] = reduce_len(session['history'], 10)

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'back':
            if session['history']:
                (label, img_id, valid) = session['history'].pop()
                if processed2processing(label, img_id, valid, session['session_uid']) :
                    session['queue'].insert(0, (
                        label,
                        img_id,
                    ))
            return redirect(url_for('index'))

        elif action in ('approve', 'reject'):
            if not session['queue']:
                return redirect(url_for('index'))

            app.logger.info(f"{'Approve' if action == 'approve' else 'Reject'} image {session['queue'][0]}")
            label, img_id = session['queue'].pop(0)
            current_img = get_status(label, img_id)

            if current_img['status'] != 'processing' or current_img['session_uid'] != session['session_uid']:
                app.logger.error("Fail to find processing image for label %s and img_id %s\n,status:%s", label, img_id, str(current_img['status']))
                return redirect(url_for('index'))

            if mark_processed(label, img_id, current_img['path'], action == 'approve'):
                session['history'].append((
                    label,
                    img_id,
                    action == 'approve',
                ))
            else:
                app.logger.error(f"Failed to mark {current_img['path']} as {'valid' if action == 'approve' else 'invalid'}")

            return redirect(url_for('index'))

    current_image = session['queue'][0] if session['queue'] else None
    if not current_image:
        return "所有图片审查完成！"

    label, img_id = current_image
    img_status = get_status(label, img_id)
    current, total = count_images()

    if img_status['path'] and os.path.exists(img_status["path"]):
        return render_template(
            'index.html',
            label=label,
            number=img_id,
            reference=urlwrapper(get_reference(label)),
            current=f"{current}/{total}",
            progress=f"{(current / total * 100):.1f}" if total else '0',
            image=urlwrapper(os.path.relpath(img_status["path"], DATA_DIR)),
            can_go_back=bool(session['history'])
        )
    else:
        if session['queue']:
            session['queue'].pop(0)
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='::', port=5000, debug=True)
