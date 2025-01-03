from flask import Flask, request, send_file, jsonify, render_template, url_for
import io
import numpy as np
from scipy.io import wavfile
import torch
from io import BytesIO
import soundfile as sf
from pydub import AudioSegment

# import whisper
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import pipeline
import module.util as util
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from module.tts import tts_male2 ,male_speaker_ids
from module.mysql import fetch_data

import whisper

# 데이터베이스에서 게이트와 푸시백 데이터를 가져옴
gates, pushBacks = fetch_data()

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

# Whisper 모델 로드
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("whisper_test2/best_model")

whisper_model = whisper.load_model('base')

# 디바이스 설정 (GPU 사용 가능 여부 확인)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def generate_speech():
    """
    텍스트를 받아 음성 파일을 생성하여 반환합니다.
    """
    if request.data:
        text_prompt = request.data.decode("utf-8")
    elif 'INPUT_TEXT' in request.form:
        text_prompt = request.form.get('INPUT_TEXT')
        
    text_prompt = util.convert_full_string(text_prompt)
    callsign = request.form.get('CALLSIGN')
    # 무작위로 ID 선택
    speaker_id = util.get_speaker_for_callsign(callsign)
    print(util.callsign_speaker_map)
    # 텍스트를 음성으로 변환
    wav = tts_male2.tts(text_prompt, speaker=speaker_id, speed=1.5)
    
    # 노이즈 추가
    noise_level = 0.005  # 노이즈 강도 조절
    wav_with_noise = util.add_noise(wav, noise_level)

    # rate = int(request.form.get('INPUT_RATE', 48500))
    rate = 22050

    wav_array = np.array(wav_with_noise)
    edited_wav_int16 = (wav_array * 32767).astype(np.int16)
    
    wav_file = io.BytesIO()
    wavfile.write(wav_file, rate, edited_wav_int16)
    wav_file.seek(0)
    
    response = send_file(wav_file, mimetype='audio/wav')
    return response

# PushBackList 인스턴스 생성
wait_pushback_lists: List[PushBackList] = []

def create_pushback_list(callsign: str, gate: int, pushbacks: List[PushBack]):
    """
    새로운 푸시백 리스트를 생성하여 대기 리스트에 추가합니다.
    """
    new_pushback_list = PushBackList(callsign=callsign, gate=gate)
    for pb in pushbacks:
        new_pushback_list.add_pushback(pb)
    wait_pushback_lists.append(new_pushback_list)

# 푸시백 리스트에서 특정 callsign을 가진 항목 제거
def remove_pushback_list_by_callsign(callsign: str):
    """
    주어진 콜사인을 가진 푸시백 리스트를 제거합니다.
    """
    global wait_pushback_lists
    wait_pushback_lists = [pb_list for pb_list in wait_pushback_lists if pb_list.callsign != callsign]
    print("푸시백 시작")

@app.route('/generate_txt', methods=['POST'])
def generate_txt():
    """
    업로드된 음성 파일을 텍스트로 변환하여 반환합니다.
    """
    try:
        wav_file = request.files['wav']
        wav_bytes = wav_file.read()  # FileStorage 객체의 데이터를 바이트로 읽음
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")  # 메모리 내에서 로드

        # 오디오 파일 로드
        # audio = AudioSegment.from_wav(temp_path)

        # 16kHz로 변환
        audio = audio.set_frame_rate(16000).set_channels(1)

        # numpy 배열로 변환
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        # 입력 특성 추출
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features

        # 입력 특성을 GPU로 이동
        input_features = input_features.to(device)

        # 모델 추론
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        # 텍스트로 변환
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # FileStorage 객체를 읽어서 numpy 배열로 변환
        # wav_data, sample_rate = sf.read(BytesIO(wav_file.read()))

        # result = pipe(wav_data)['text']
        
        result = transcription

        org_result = whisper_model.transcribe(audio_array, fp16=True)['text']
        print(f"순정 Whisper : {org_result}")

        # print(result)
        # print(f"기존 Whisper : {org_result}")
        print(f"파인튜닝 후 Whisper : {result}")

        original_text = result.upper().replace('-', '').replace(',', '')

        converted_text = util.reverse_convert_full_string(original_text)

        callsign, acts, approved, rwy, gate = util.extract_callsign_and_act(converted_text)

        return jsonify({
            'converted_text': converted_text,
            'callsign': callsign,
            'acts': acts,
            'b_approved':approved,
            'rwy':rwy,
            'gate':gate
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({
            'converted_text': "오류",
            'callsign': "",
            'acts': [],
            'b_approved':''
        })

# 푸시백 요청 대기
@app.route('/request_pushback', methods=['POST'])
def request_pushback():
    """
    푸시백 요청을 받아 대기 리스트에 추가합니다.
    """
    callsign = request.form.get('callsign', '0')
    gate = int(request.form.get('gate', 0))

    # 푸시백 리스트에서 게이트별 푸시백으로 필터링
    filtered_pushbacks = [pb for pb in pushBacks if pb.gate == gate]

    create_pushback_list(callsign, gate, filtered_pushbacks)

    return jsonify({
            "message": "Pushback list added successfully."
        })

if __name__ == "__main__":
    config = util.load_config('config.ini')
    server_config = config['ServerConfig']
    app.run(host='0.0.0.0', port=int(server_config['port']), threaded=True)