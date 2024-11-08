from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os

# Flask 애플리케이션 설정
app = Flask(__name__)
app.secret_key = "supersecretkey"

# 파일 저장 경로 설정
UPLOAD_FOLDER_AUDIO = 'dataset/audio'
UPLOAD_FOLDER_TEXT = 'dataset/text'
os.makedirs(UPLOAD_FOLDER_AUDIO, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TEXT, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 음성 파일 처리
        if 'audio' not in request.files or request.files['audio'].filename == '':
            flash('No audio file provided', 'error')
            return redirect(request.url)
        
        audio_file = request.files['audio']
        audio_filename = audio_file.filename
        audio_path = os.path.join(UPLOAD_FOLDER_AUDIO, audio_filename)
        audio_file.save(audio_path)
        
        # 텍스트 처리
        transcript = request.form['transcript']
        if transcript.strip() == '':
            flash('No transcript provided', 'error')
            return redirect(request.url)
        
        text_filename = os.path.splitext(audio_filename)[0] + '.txt'
        text_path = os.path.join(UPLOAD_FOLDER_TEXT, text_filename)
        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(transcript)
        
        flash('File and transcript saved successfully', 'success')
        return redirect(url_for('maker', filename=audio_filename))
    
    # 업로드된 파일의 이름을 쿼리 파라미터에서 가져오기
    filename = request.args.get('filename')
    return render_template('maker.html', filename=filename)

@app.route('/audio/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER_AUDIO, filename)

if __name__ == '__main__':
    app.run(debug=True)
