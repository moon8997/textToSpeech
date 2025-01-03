import re
import configparser
from difflib import SequenceMatcher
import random
import numpy as np

from module.tts import male_speaker_ids
from .dict import airline, pronunciation, acts, number_map

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def convert_airline_code(word):
    """
    항공사 코드를 변환합니다.

    Args:
        word (str): 변환할 항공사 코드와 비행기 번호

    Returns:
        str: 변환된 항공사 이름과 비행기 번호 발음
    """

    if word[:3].isalpha():
        airline_code = word[:3]
        flight_number = word[3:]
    else:
        airline_code = word[:2]
        flight_number = word[2:]

    airline_name = airline.get(airline_code, airline_code)
    flight_pronunciation = ''.join(pronunciation.get(char.upper(), char) for char in flight_number)
    return f'{airline_name} {flight_pronunciation}'

def convert_digit_to_string(word):
    """
    숫자를 문자열로 변환합니다.

    Args:
        word (str): 변환할 숫자 문자열

    Returns:
        str: 변환된 문자열
    """
    return ' '.join(pronunciation.get(char, char) for char in word)

def convert_full_string(input_string):
    """
    입력된 문자열을 특정 형식으로 변환합니다.

    Args:
        input_string (str): 변환할 문자열

    Returns:
        str: 변환된 문자열
    """
    upper_string = input_string.upper().replace('&', ' AND ')
    input_string = upper_string
    result = []
    words = re.findall(r'\b\w+\b|\S', input_string)

    for i, word in enumerate(words):
        if word.startswith('FC') and word[2:].isdigit():
            result.append('flight check ' + convert_digit_to_string(word[2:]))
        elif len(word) > 2 and word[:2].isalpha() and word[2:].isdigit() or len(word) > 3 and word[:3].isalpha() and word[3:].isdigit():
            result.append(convert_airline_code(word))
        elif word[-1] in ['R', 'L'] and word[:-1].isdigit():
            result.append(convert_digit_to_string(word[:-1]) + (' right' if word[-1] == 'R' else ' left'))
        elif any(char.isdigit() for char in word) and any(char.isalpha() for char in word) or word.isdigit() or len(word) == 1:
            result.append(convert_digit_to_string(word))
        elif word == "RWY":
            result.append("RUNWAY")
        else:
            result.append(word)

        if i < len(words) - 1:
            result.append(' ')

    return ' '.join(result)

def reverse_convert_full_string(input_string):
    """
    변환된 문자열을 원래 형식으로 되돌립니다.

    Args:
        input_string (str): 변환된 문자열

    Returns:
        str: 원래 형식으로 되돌린 문자열
    """
    arr = []  # 반환된 숫자와 단어를 담을 리스트
    combined_number = ""  # 숫자를 결합할 변수
    is_runway_mode = False  # RUNWAY 모드인지 확인하는 플래그

    # 문장 부호를 처리하기 위해 단어와 문장 부호를 분리
    words = re.findall(r'\b\w+\b|[.,!?;]', input_string)

    for word in words:
        upper_word = word.upper()  # 단어를 대문자로 변환

        if upper_word in number_map:  # 숫자 단어인 경우
            combined_number += number_map[upper_word]  # 숫자를 결합
        elif upper_word == "RUNWAY":  # RUNWAY를 RWY로 변환
            is_runway_mode = True
            arr.append("RWY")
        elif upper_word in ["RIGHT", "LEFT"]:  # 방향을 R 또는 L로 변환
            if is_runway_mode:  # RUNWAY 모드인 경우에만 변환
                arr[-1] += combined_number + ("R" if upper_word == "RIGHT" else "L")  # RWY와 숫자, 방향 결합
                combined_number = ""  # 초기화
                is_runway_mode = False  # RUNWAY 모드 종료
            else:
                arr.append(upper_word)  # RUNWAY와 함께 나오지 않으면 변환하지 않음
        else:
            if combined_number:  # 결합된 숫자가 있으면
                if is_runway_mode:  # RUNWAY 모드인 경우
                    arr[-1] += combined_number  # RWY와 숫자 결합
                    combined_number = ""  # 초기화
                    is_runway_mode = False  # RUNWAY 모드 종료
                else:
                    arr.append(combined_number)  # 아니면 숫자만 추가
                    combined_number = ""  # 초기화
            arr.append(word)  # 숫자가 아니면 그대로 추가

    if combined_number:  # 마지막에 남은 결합된 숫자 추가
        arr.append(combined_number)

    converted_text = ' '.join(arr)

    # 항공사 코드 변환
    for key, value in airline.items():
        converted_text = re.sub(r'\b' + value.upper() + r'\b', key, converted_text)

    # 항공사 코드와 숫자 결합
    words = converted_text.split()
    final_result = []
    i = 0
    while i < len(words):
        if words[i] in airline.keys() and i + 1 < len(words) and words[i + 1].isdigit():
            final_result.append(words[i] + words[i + 1])
            i += 2
        else:
            final_result.append(words[i])
            i += 1

    # 문장 부호 앞뒤의 공백 제거
    final_text = ' '.join(final_result)
    final_text = re.sub(r'\s+([.,!?;])', r'\1', final_text)

    return final_text


reverse_pronunciation = {v.strip(): k for k, v in pronunciation.items()}

def extract_callsign_and_act(text):
    """
    주어진 텍스트에서 콜사인과 행위를 추출합니다.

    Args:
        text (str): 분석할 텍스트

    Returns:
        tuple: 콜사인, 추출된 행위 리스트, 승인 여부, 활주로, 게이트
    """
    callsign = ''
    extracted_acts = []
    approved = False
    rwy = ''
    gate = ''

    callsign_pattern = re.compile(r'\b(' + '|'.join(airline.keys()) + r')\d+\b')
    callsign_match = callsign_pattern.search(text)
    if callsign_match:
        callsign = callsign_match.group(0)
    
    # acts를 정렬하여 더 긴 문구가 먼저 매치되도록 함
    sorted_acts = sorted(acts, key=len, reverse=True)
    act_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_acts)) + r')\b', re.IGNORECASE)
    act_matches = act_pattern.findall(text)
    if act_matches:
        extracted_acts = [act.upper() for act in act_matches]
    
    # APPROVED 확인
    if 'APPROVED' in text.upper():
        approved = True

    # RWY(활주로) 추출
    rwy_pattern = re.compile(r'\bRWY\s*(\d{1,2}[LCR]?)\b', re.IGNORECASE)
    rwy_match = rwy_pattern.search(text)
    if rwy_match:
        rwy = rwy_match.group(1)

    # GATE 추출
    gate_pattern = re.compile(r'\bGATE\s*(\d+)\b', re.IGNORECASE)
    gate_match = gate_pattern.search(text)
    if gate_match:
        gate = gate_match.group(1)

    return callsign, extracted_acts, approved, rwy, gate


# 콜사인과 화자 ID 매핑을 저장할 딕셔너리
callsign_speaker_map = {}

def get_speaker_for_callsign(callsign):
    """
    주어진 콜사인에 대한 화자 ID를 반환합니다.

    Args:
        callsign (str): 콜사인

    Returns:
        int: 화자 ID
    """
    # 콜사인이 이미 매핑되어 있는지 확인
    if callsign in callsign_speaker_map:
        return callsign_speaker_map[callsign]
    else:
        # 무작위로 화자 ID 선택
        random_speaker_id = random.choice(male_speaker_ids)
        # 매핑 저장
        callsign_speaker_map[callsign] = random_speaker_id
        return random_speaker_id
    

def add_noise(wav, noise_level):
    """
    음성 데이터에 노이즈를 추가하는 함수
    :param wav: 원본 음성 데이터
    :param noise_level: 노이즈의 강도 (0과 1 사이의 값)
    :return: 노이즈가 추가된 음성 데이터
    """
    noise = np.random.randn(len(wav))  # 정규 분포를 따르는 노이즈 생성
    wav_with_noise = wav + noise_level * noise  # 원본 음성에 노이즈 추가
    wav_with_noise = np.clip(wav_with_noise, -1, 1)  # 값의 범위를 -1과 1 사이로 제한
    return wav_with_noise