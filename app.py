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
from difflib import SequenceMatcher
from module.tts import tts_male2 ,male_speaker_ids
from module.mysql import fetch_data

import whisper
import re

# 텍스트 파일을 리스트로 읽어오기
with open('controller_script.txt', 'r', encoding='utf-8') as file:
    script_lines = [line.strip() for line in file if line.strip()]  # 빈 줄 제거

def calculate_similarity(a, b):
    """
    두 문자열 간 유사도를 계산합니다 (0~1 사이의 값 반환).
    """
    return SequenceMatcher(None, a, b).ratio()

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

whisper_model = whisper.load_model('small')

# 디바이스 설정 (GPU 사용 가능 여부 확인)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = Flask(__name__)

callsign_list = []

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

    global callsign_list

    text_prompt = util.convert_full_string(text_prompt)
    callsign = request.form.get('CALLSIGN').upper()

    if callsign:
        if callsign not in callsign_list:
            callsign_list.append(callsign)
        
    # 디버깅용 출력
    print("Current callsign_list:", callsign_list)

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


def find_most_similar_callsign(input_callsign, callsign_list):
    """가장 유사한 콜사인을 콜사인 리스트에서 찾습니다."""
    max_similarity = 0
    most_similar_callsign = None

    for callsign in callsign_list:
        similarity = calculate_similarity(input_callsign, callsign)
        print(similarity)
        if similarity > max_similarity and similarity > 0.55 :
            max_similarity = similarity
            most_similar_callsign = callsign
    print(f"콜사인 max_similarity : {max_similarity}")
    return most_similar_callsign

def remove_parentheses_content(text):
    """
    텍스트에서 괄호와 그 안의 내용을 제거합니다.
    """
    return re.sub(r'\([^)]*\)', '', text).strip()

def replace_callsign(original_line, new_callsign):
    """
    대본의 콜사인을 새로운 콜사인으로 대체하고, 괄호를 제거합니다.
    """
    # 콜사인 대체
    updated_line = re.sub(r'\b[A-Z]{3}\d{3,4}\b', new_callsign, original_line)
    # 괄호만 제거 (내용은 유지)
    return re.sub(r'[()]', '', updated_line).strip()

def extract_first_number(text):
    current_number = ""
    
    for char in text:
        if char.isdigit():
            # 숫자일 경우 current_number에 추가
            current_number += char
        else:
            if current_number:
                # 숫자가 끝나면 첫 번째 숫자 반환
                return current_number
    # 마지막 숫자가 있을 경우 반환
    if current_number:
        return current_number
    return None  # 숫자가 없을 경우

@app.route('/generate_txt', methods=['POST'])
def generate_txt(first_time: int = 0):
    """
    업로드된 음성 파일을 텍스트로 변환하여 반환합니다.
    """
    try:
        if first_time == 1:
            with open('static/voice/start.wav', 'rb') as f:
                wav_bytes = f.read()
        else:
            # 음성 파일을 읽어서 numpy 배열로 변환
            wav_file = request.files['wav']
            wav_bytes = wav_file.read()  # FileStorage 객체의 데이터를 바이트로 읽음

        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")  # 메모리 내에서 로드

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
            predicted_ids = model.generate(
                input_features,
                max_length=100,  # 최대 출력 길이 제한
                temperature=0.7,  # 다양성을 증가시키기 위해 약간 높임
                repetition_penalty=1.2,  # 반복 패턴 억제
                no_repeat_ngram_size=3,  # 반복 n-gram 방지
                num_beams=5  # 빔 서치를 활용하여 최적의 결과 탐색
            )

        # 텍스트로 변환
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        org_result = whisper_model.transcribe(audio_array, fp16=True)['text']

        whisper_text = org_result.upper().replace('-', '').replace(',', '')

        converted_whisper_text = util.reverse_convert_full_string(whisper_text)
        print(f"파인튜닝 전 Whisper : {converted_whisper_text}")

        whisper_callsine_number = extract_first_number(converted_whisper_text.replace(' ', ''))

        if first_time != 1:
            result = transcription

            print(f"파인튜닝 후 Whisper : {result}")

            original_text = result.upper().replace('-', '').replace(',', '')

            converted_text = util.reverse_convert_full_string(original_text)

            callsign, acts, approved, rwy, gate = util.extract_callsign_and_act(converted_text)
            # 콜사인 추출
            # callsign과 whisper_callsine_number 유효성 확인
            if callsign or whisper_callsine_number:  # 둘 중 하나라도 유효하면 실행
                input_callsign = (callsign or '') + (whisper_callsine_number or '')  # None이면 빈 문자열로 대체
            else:
                input_callsign = None  # 둘 다 None이면 None 설정

            # 유사도가 60% 이상인 줄 찾기
            matched_line = None
            max_similarity = 0
            similarity_threshold = 0.5

            print(f"converted_text: {converted_text}")

            most_similar_callsign = None

            for line in script_lines:
                line_cleaned = remove_parentheses_content(line)
                similarity = calculate_similarity(converted_text.strip().lower(), line_cleaned.lower())
                # print(f"유사도: {similarity}")
                if similarity > similarity_threshold and similarity > max_similarity:
                    matched_line = line
                    max_similarity = similarity

            if matched_line:
                # 콜사인 대체
                print(f"입력 콜사인: {input_callsign}")
                most_similar_callsign = find_most_similar_callsign(input_callsign, callsign_list)
                if most_similar_callsign:
                    result = replace_callsign(matched_line, most_similar_callsign)
                else:
                    result = matched_line

                print(f"최대 유사도: {max_similarity}")
            else:
                result = '일치하는 대사가 없습니다.'

            print(f"리턴 : {result}")

            return jsonify({
                'converted_text': result,
                'callsign': most_similar_callsign,
                'acts': acts,
                'b_approved': approved,
                'rwy': rwy,
                'gate': gate
            })
        else:
            print("서버시작")
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({
            'converted_text': "오류",
            'callsign': "",
            'acts': [],
            'b_approved': ''
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
    
    # 서버가 완전히 켜지기 전에 초기 작업 수행
    from threading import Thread
    
    def run_generate_txt():
        try:
            print("서버 초기화 작업 시작")
            generate_txt(1)
        except Exception as e:
            print(f"초기화 작업 중 오류 발생: {e}")
    
    # 초기 작업 실행 (백그라운드에서 실행)
    initialization_thread = Thread(target=run_generate_txt)
    initialization_thread.start()
    
    # Flask 앱 실행
    app.run(host='0.0.0.0', port=int(server_config['port']), threaded=True)