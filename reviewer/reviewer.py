import os
from flask import Flask, render_template, request, session, redirect, url_for
from datetime import datetime

app = Flask(__name__, static_folder='data')
app.secret_key = os.urandom(24)
DATA_DIR = 'data'

def get_all_pending():
    pending_images = []
    for label in os.listdir(DATA_DIR):
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.pending'):
                    number = file.split('.')[0]
                    if number == '1':
                        continue
                    img_path = os.path.join(label_dir, file)
                    pending_images.append((label, number, img_path))
    return sorted(pending_images, key=lambda x: (x[0], int(x[1])))

def init_user_session():
    # 使用普通列表替代deque
    if 'queue' not in session:
        session['queue'] = get_all_pending()
    if 'processed' not in session:
        session['processed'] = []
    if 'history' not in session:
        session['history'] = []
    # 转换列表为可序列化格式
    session.modified = True

def get_reference(label):
    return os.path.join(label, '1.jpg')

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
                
                # 在列表头部插入回退的图片
                session['queue'].insert(0, (
                    last_action['label'], 
                    last_action['number'],
                    new_path
                ))
                
                # 从已处理队列移除
                if session['processed']:
                    session['processed'].pop()
            
            return redirect(url_for('index'))
            
        elif action in ('approve', 'reject'):
            if not session['queue']:
                return redirect(url_for('index'))
            
            # 获取当前图片（列表第一个元素）
            current_image = session['queue'][0]
            label, number, img_path = current_image
            new_ext = '.jpg' if action == 'approve' else '.invalid'
            new_path = f"{os.path.splitext(img_path)[0]}{new_ext}"
            
            try:
                os.rename(img_path, new_path)
                # 记录操作历史
                session['history'].append({
                    'label': label,
                    'number': number,
                    'original_path': img_path,
                    'new_path': new_path,
                    'action': action,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                # 移动图片到已处理队列
                session['processed'].append(session['queue'].pop(0))
            except Exception as e:
                print(f"Error processing image: {str(e)}")
            
            return redirect(url_for('index'))

    # 获取当前图片
    current_image = session['queue'][0] if session['queue'] else None
    
    if not current_image:
        return "所有图片审查完成！"
    
    # 计算进度
    label, number, img_path = current_image
    current, total = count_images()
    
    return render_template(
        'index.html',
        label=label,
        number=number,
        reference=get_reference(label),
        current=f"{current}/{total}",
        progress=f"{(current/total*100):.1f}" if total else '0',
        image=os.path.relpath(img_path, DATA_DIR),
        can_go_back=bool(session['history'])
    )

if __name__ == '__main__':
    app.run(threaded=True)
