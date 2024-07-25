from flask import Flask, request, send_file, jsonify, render_template
import io
import numpy as np
from scipy.io import wavfile
import torch
import whisper

from module.tts import tts_female, tts_male
from module.util import convert_full_string, reverse_convert_full_string, extract_callsign_and_act, load_config
from module.mysql import fetch_data

gates, pushBacks = fetch_data() # DB 

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model
whisper_model = whisper.load_model('base')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def generate_speech():
    if request.data:
        text_prompt = request.data.decode("utf-8")
    elif 'INPUT_TEXT' in request.form:
        text_prompt = request.form.get('INPUT_TEXT')

    text_prompt = convert_full_string(text_prompt)
    type = request.form.get('TYPE', '0')

    if type == '1':
        wav = tts_male.tts(text_prompt, speaker_wav="./voice/chim.wav", language="en")
    else:
        wav = tts_female.tts(text_prompt)

    rate = int(request.form.get('INPUT_RATE'))

    wav_array = np.array(wav)
    edited_wav_int16 = (wav_array * 32767).astype(np.int16)
    
    wav_file = io.BytesIO()
    wavfile.write(wav_file, rate, edited_wav_int16)
    wav_file.seek(0)
    
    response = send_file(wav_file, mimetype='audio/wav')
    return response

@app.route('/generate_txt', methods=['POST'])
def generate_txt():
    try:
        wav_file = request.files['wav']
        temp_path = "temp.wav"
        wav_file.save(temp_path)

        result = whisper_model.transcribe(temp_path)
        original_text = result['text'].upper().replace('-', '').replace(',', '')
        converted_text = reverse_convert_full_string(original_text)
        callsign, act = extract_callsign_and_act(converted_text)

        gate = 100003

        # gate가 100003인 항목 필터링
        filtered_pushbacks = [pb for pb in pushBacks if pb.gate == gate]

        # print(filtered_pushbacks)

        return jsonify({
            'converted_text': converted_text,
            'callsign': callsign,
            'act': act
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({
            'converted_text': "오류",
            'callsign': "",
            'act': ""
        })



if __name__ == "__main__":
    config = load_config('config.ini')
    server_config = config['ServerConfig']
    app.run(host='0.0.0.0', port=int(server_config['port']), threaded=True)
