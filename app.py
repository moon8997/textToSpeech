from flask import Flask, request, send_file, jsonify, render_template, url_for
import io
import numpy as np
from scipy.io import wavfile
import torch
import whisper
from dataclasses import dataclass, field
from typing import List, Optional

from module.tts import tts_female, tts_male
from module.util import convert_full_string, reverse_convert_full_string, extract_callsign_and_act, load_config
from module.mysql import fetch_data

gates, pushBacks = fetch_data() # DB 

@dataclass
class PushBack:
    pushback_seq: int
    gate: int
    pushback: str
    text: str

@dataclass
class PushBackList:
    callsign: int
    gate: int
    pushbacks: List[PushBack] = field(default_factory=list)

    def add_pushback(self, pushback: PushBack):
        self.pushbacks.append(pushback)

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

    rate = int(request.form.get('INPUT_RATE', 48500))

    wav_array = np.array(wav)
    edited_wav_int16 = (wav_array * 32767).astype(np.int16)
    
    wav_file = io.BytesIO()
    wavfile.write(wav_file, rate, edited_wav_int16)
    wav_file.seek(0)
    
    response = send_file(wav_file, mimetype='audio/wav')
    return response

# PushBackList 인스턴스 생성
wait_pushback_lists: List[PushBackList] = []

def create_pushback_list(callsign: str, gate: int, pushbacks: List[PushBack]):
    new_pushback_list = PushBackList(callsign=callsign, gate=gate)
    for pb in pushbacks:
        new_pushback_list.add_pushback(pb)
    wait_pushback_lists.append(new_pushback_list)

# 푸시백 리스트에서 특정 callsign을 가진 항목 제거
def remove_pushback_list_by_callsign(callsign: str):
    global wait_pushback_lists
    wait_pushback_lists = [pb_list for pb_list in wait_pushback_lists if pb_list.callsign != callsign]
    print("푸시백 시작")

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


        if act.replace(' ', '') == 'PUSHBACK' and callsign:

            pushBack: Optional[PushBackList] = next((pb_list for pb_list in wait_pushback_lists if pb_list.callsign == callsign), None)
            print(original_text)
            if pushBack:
                pushback_texts = [pushback.text for pushback in pushBack.pushbacks]

                if any(text in original_text.replace(' ', '') for text in pushback_texts):
                    remove_pushback_list_by_callsign(callsign)
                else:
                    return jsonify({
                        'converted_text': converted_text,
                        'callsign': callsign,
                        'act': act,
                        'url': url_for('static', filename='voice/invalid-pushback.mp3')
                    })
            else:
                return jsonify({
                    'converted_text': converted_text,
                    'callsign': callsign,
                    'act': act,
                    'url': url_for('static', filename='voice/invalid-callsign.mp3')
                })
        return jsonify({
            'converted_text': converted_text,
            'callsign': callsign,
            'act': act,
            'url': ""
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({
            'converted_text': "오류",
            'callsign': "",
            'act': "",
            'url': ""
        })

# 푸시백 요청 대기
@app.route('/request_pushback', methods=['POST'])
def request_pushback():
    callsign = request.form.get('callsign', '0')
    gate = int(request.form.get('gate', 0))

    # 푸시백 리스트에서 게이트별 푸시백으로 필터링
    filtered_pushbacks = [pb for pb in pushBacks if pb.gate == gate]

    create_pushback_list(callsign, gate, filtered_pushbacks)

    return jsonify({
            "message": "Pushback list added successfully."
        })

if __name__ == "__main__":
    config = load_config('config.ini')
    server_config = config['ServerConfig']
    app.run(host='0.0.0.0', port=int(server_config['port']), threaded=True
            # , debug=True
            )
