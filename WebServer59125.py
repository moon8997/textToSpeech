from flask import Flask, request, send_file, Response, render_template, redirect, url_for
from TTS.api import TTS
import torch
import whisper
import numpy as np
from scipy.io import wavfile
import configparser
import io
from dict import airline, pronunciation
import re

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config['ServerConfig']

tts_female = TTS("tts_models/en/jenny/jenny").to(device)
tts_male = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device) # 퀄리티가 별론데 속도가 빠름
whisper_model = whisper.load_model('base')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def generate_speech():
    # data = request.get_json()
    # callsign = data['callsign']
    # text_prompt = data['input']
    # type = data['type']

    # request.data가 비어 있지 않다면, 이를 디코딩하여 text_prompt로 사용합니다.
    if request.data:
        text_prompt = request.data.decode("utf-8")
    # 그렇지 않고 request.form에 'INPUT_TEXT' 필드가 있다면, 이를 text_prompt로 사용합니다.
    elif 'INPUT_TEXT' in request.form:
        text_prompt = request.form.get('INPUT_TEXT')
    
    text_prompt = convert_full_string(text_prompt)
    
    # print(text_prompt.upper().replace('  ', ' ').replace(' .', '.').replace(' ,' , ','))

    # 'TYPE'이 없을 경우 기본값으로 '0'을 사용합니다.
    type = request.form.get('TYPE', '0')

    if(type == '1') :
        wav = tts_male.tts(text_prompt, speaker_wav="./voice/man2.mp3", language="en")
        rate = 16000
    else :
        wav = tts_female.tts(text_prompt)
        rate = 48000
        
    
    wav_array = np.array(wav)
    # 16비트 정수로 변환
    edited_wav_int16 = (wav_array * 32767).astype(np.int16)
    
    # wav 파일을 메모리에 저장하기 위한 BytesIO 객체 생성
    wav_file = io.BytesIO()

    wavfile.write(wav_file, rate, edited_wav_int16)
    
    # BytesIO 객체를 파일 처럼 다룰 수 있게 하기 위해 seek 메소드를 사용
    wav_file.seek(0)
    
    # Response 객체 생성
    response = send_file(wav_file, mimetype='audio/wav')
    # Response 헤더에 callsign 추가
    # response.headers['Callsign'] = callsign

    return response

@app.route('/generate_txt', methods=['POST'])
def generate_txt():
    wav_file = request.files['wav']
    
    temp_path = "temp.wav"
    wav_file.save(temp_path)

    result = whisper_model.transcribe(temp_path)
    # os.remove(temp_path)

    return result['text']

def convert_airline_code(word):
    # 알파벳이 세 자리인 경우
    if word[:3].isalpha():
        airline_code = word[:3]
        flight_number = word[3:]
    # 알파벳이 두 자리인 경우
    else:
        airline_code = word[:2]
        flight_number = word[2:]

    # 항공사 코드를 항공사 이름으로 변환
    airline_name = airline.get(airline_code, airline_code)

    # 비행기 번호를 발음으로 변환
    flight_pronunciation = ''.join(pronunciation.get(char.upper(), char) for char in flight_number)

    return f'{airline_name} {flight_pronunciation}'

def convert_digit_to_string(word):
    # 숫자를 발음으로 변환
    return ' '.join(pronunciation.get(char, char) for char in word)

def convert_full_string(input_string):
    input_string = input_string.upper().replace('&', ' AND ')
    
    result = []

    # 문자열을 단어와 특수 문자로 분리
    words = re.findall(r'\b\w+\b|\S', input_string)

    for i, word in enumerate(words):
        # 단어가 알파벳 2~3글자 + 숫자 형태라면 항공사 코드로 처리
        if word.startswith('FC') and word[2:].isdigit():
            result.append('flight check ' + convert_digit_to_string(word[2:]))
        elif len(word) > 2 and word[:2].isalpha() and word[2:].isdigit() or len(word) > 3 and word[:3].isalpha() and word[3:].isdigit():
            result.append(convert_airline_code(word))
        elif word[-1] in ['R', 'L'] and word[:-1].isdigit():
            result.append(convert_digit_to_string(word[:-1]) + ('right' if word[-1] == 'R' else 'left'))
        # 알파벳과 숫자가 섞인 경우 처리 (예: A15, 15A 등)
        elif any(char.isdigit() for char in word) and any(char.isalpha() for char in word) or word.isdigit() or len(word) == 1:
            result.append(convert_digit_to_string(word))
        else:
            result.append(word)

        # 띄어쓰기 처리 (마지막 단어 제외)
        if i < len(words) - 1: 
            result.append(' ')

    return ''.join(result)


if __name__ == "__main__":
    config_data = load_config('config.ini')
    app.run(host='0.0.0.0', port=int(config_data['port']), threaded=True)