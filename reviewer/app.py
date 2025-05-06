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


def get_one_pending(session_uid):
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


threading.Thread(target=release_processing_images, daemon=True).start()


def init_user_session():
    if 'queue' not in session:
        session['queue'] = []
    if 'processed' not in session:
        session['processed'] = []
    if 'history' not in session:
        session['history'] = []
    if 'session_uid' not in session:
        session['session_uid'] = str(uuid.uuid4())

    if not session['queue']:
        result = get_one_pending(session['session_uid'])
        if result:
            label, number, img_path = result
            session['queue'].append((label, number, img_path))

    session.modified = True


def get_reference(label):
    ref_path = os.path.join(DATA_DIR, label, '1.jpg')
    if os.path.exists(ref_path):
        return os.path.relpath(ref_path, DATA_DIR)
    return None

def count_images():
    valid = invalid = pending = 0
    for label in os.listdir(DATA_DIR):
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file == '1.jpg':
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

@app.route('/', methods=['GET', 'POST'])
def index():
    init_user_session()

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'back':
            if session['history']:
                last_action = session['history'].pop()
                old_path = last_action['new_path']
                new_path = last_action['original_path']
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                session['queue'].insert(0, (
                    last_action['label'],
                    last_action['number'],
                    new_path
                ))
                if session['processed']:
                    session['processed'].pop()
            return redirect(url_for('index'))

        elif action in ('approve', 'reject'):
            if not session['queue']:
                return redirect(url_for('index'))

            current_image = session['queue'][0]
            label, number, img_path = current_image
            filename = os.path.basename(img_path)
            parts = filename.split('.')

            if len(parts) < 4 or parts[-1] != 'processing':
                return redirect(url_for('index'))

            stored_uid = parts[-3]
            if stored_uid != session['session_uid']:
                return redirect(url_for('index'))

            # 提取原始文件名（带扩展名）
            base_parts = parts[:-3]  # 保留前面部分，如 ['123', 'jpg']
            base_name = '.'.join(base_parts)  # '123.jpg'

            new_ext = '.jpg' if action == 'approve' else '.invalid'
            new_path = os.path.join(os.path.dirname(img_path), base_name + new_ext)

            try:
                os.rename(img_path, new_path)
                session['history'].append({
                    'label': label,
                    'number': number,
                    'original_path': img_path,
                    'new_path': new_path,
                    'action': action,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                session['processed'].append(session['queue'].pop(0))
            except Exception as e:
                print(f"Error processing image: {str(e)}")

            return redirect(url_for('index'))

    current_image = session['queue'][0] if session['queue'] else None
    if not current_image:
        return "所有图片审查完成！"

    label, number, img_path = current_image
    current, total = count_images()

    if os.path.exists(img_path):
        return render_template(
            'index.html',
            label=label,
            number=number,
            reference=urlwrapper(get_reference(label)),
            current=f"{current}/{total}",
            progress=f"{(current / total * 100):.1f}" if total else '0',
            image=urlwrapper(os.path.relpath(img_path, DATA_DIR)),
            can_go_back=bool(session['history'])
        )
    else:
        if session['queue']:
            session['queue'].pop(0)
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
