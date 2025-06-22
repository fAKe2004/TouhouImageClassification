import os
from flask import Flask, request, render_template, redirect, flash
from PIL import Image
import base64
from io import BytesIO
import requests

from runtime import DEFAULT_LABEL_LANG, serve_batch, is_daemon_running, is_daemon_cuda

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = []
    lang = request.form.get('lang', DEFAULT_LABEL_LANG)
    
    if request.method == 'POST':
        images = []
        filenames = []

        # Handle file uploads
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file and file.filename:
                    images.append(Image.open(file.stream))
                    filenames.append(file.filename)

        # Handle URL uploads
        urls_string = request.form.get('urls', '')
        if urls_string:
            urls = [u.strip() for u in urls_string.splitlines()]
            for url in urls:
                if url:
                    try:
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        images.append(Image.open(response.raw))
                        filenames.append(url.split('/')[-1])
                    except Exception as e:
                        flash(f'Error fetching URL {url}: {e}', 'error')

        if not images:
            flash('No valid files or URLs were provided.', 'error')
            return redirect(request.url)

        try:

            predictions, _ = serve_batch(images, target_lang=lang)

            for i, (img, pred) in enumerate(zip(images, predictions)):
                label, confidence = pred
                buffered = BytesIO()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                prediction_str = f"{label}\n{confidence:.1%}"
                
                results.append({'image': img_str, 'prediction': prediction_str, 'filename': filenames[i]})
            
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
    

    return render_template('index.html', results=results, daemon_running=running, device_mode=device_mode, add_intro=add_intro, lang=lang)

def run_ipv4():
    app.run(host='0.0.0.0', port=20810, debug=False)

def run_ipv6():
    app.run(host='::', port=20811, debug=False) # EoSD release date


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the Flask app.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    if args.debug:
        app.run(host='0.0.0.0', port=80, debug=True)
    else:
        import threading
        t1 = threading.Thread(target=run_ipv4)
        t2 = threading.Thread(target=run_ipv6)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()