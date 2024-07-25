import re
import configparser
from difflib import SequenceMatcher

from .dict import airline, pronunciation, acts

def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def convert_airline_code(word):
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
    return ' '.join(pronunciation.get(char, char) for char in word)

def convert_full_string(input_string):
    input_string = input_string.upper().replace('&', ' AND ')
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

    return ''.join(result)

def reverse_convert_full_string(input_string):
    for key, value in airline.items():
        input_string = input_string.replace(value.upper(), key)
    # for key, value in airline.items():
    #     ratio = similar(word, value.upper())
    #     if ratio > highest_ratio and ratio >= 0.8:
            
    words = re.findall(r'\b\w+\b|\S', input_string)
    result = []
    skip_next = False

    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            result.append(' ')
            continue
        if word in airline:
            next_word = words[i + 1] if i + 1 < len(words) else ""
            if next_word.isdigit():
                result.append(word + next_word)
                skip_next = True
            else:
                result.append(word)
        else:
            if word in reverse_pronunciation:
                result.append(reverse_pronunciation[word])
            else:
                result.append(word)
        
        if i < len(words) - 1 and not skip_next:
            result.append(' ')

    final_result = ''.join(result).strip()
    return final_result

reverse_pronunciation = {v.strip(): k for k, v in pronunciation.items()}

def extract_callsign_and_act(text):
    callsign = ''
    act = ''
    
    callsign_pattern = re.compile(r'\b(' + '|'.join(airline.keys()) + r')\d+\b')
    callsign_match = callsign_pattern.search(text)
    if callsign_match:
        callsign = callsign_match.group(0)
    
    act_pattern = re.compile(r'\b(' + '|'.join(acts) + r')\b', re.IGNORECASE)
    act_match = act_pattern.search(text)
    if act_match:
        act = act_match.group(0).upper().replace(' ', '')
    
    return callsign, act