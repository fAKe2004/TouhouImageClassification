import os
from flask import Flask, request, render_template, redirect, flash
from PIL import Image
import base64
from io import BytesIO

from runtime import serve_batch, is_daemon_running, is_daemon_cuda, check_cuda_available

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = []
    
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        try:
            images = [Image.open(file.stream) for file in files]
            predictions, _ = serve_batch(images)

            for i, (img, pred) in enumerate(zip(images, predictions)):
                label, confidence = pred
                buffered = BytesIO()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                prediction_str = f"{label}\n{confidence:.1%}"
                
                results.append({'image': img_str, 'prediction': prediction_str, 'filename': files[i].filename})
            
            flash(f'Classification complete for {len(results)} image(s)!', 'info')
              
        except Exception as e:
            flash(f'An error occurred: {e}', 'error')
            return redirect(request.url)

    device_mode = None
    running = is_daemon_running()
    if running:
        device_mode = "CUDA" if is_daemon_cuda() else "CPU"
    else:
        device_mode = "Inactive"
    add_intro = not results
    

    return render_template('index.html', results=results, daemon_running=running, device_mode=device_mode, add_intro=add_intro)

if __name__ == '__main__':
    app.run(debug=True)