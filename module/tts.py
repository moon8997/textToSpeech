from TTS.api import TTS
import torch

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# print(TTS().list_models())

# Load TTS models
# tts_female = 'TTS("tts_models/en/jenny/jenny").to(device)'
# tts_male = 'TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)'  
# tts_male2 = TTS("tts_models/en/vctk/fast_pitch").to(device) 
tts_male2 = TTS("tts_models/en/vctk/vits").to(device) 


# 남성 목록
male_speaker_ids = [
    'p226', 'p228', 'p230', 'p232', 'p233', 
    'p234', 'p238', 'p241', 'p251', 'p252', 
    'p253', 'p254', 'p256', 'p258', 'p260', 'p262', 
    'p264', 'p265', 'p266', 'p267', 'p274',
    'p279', 'p281', 'p285', 'p286', 'p287', 'p298', 
    'p299', 'p301', 'p302', 'p304', 'p307', 'p308', 
    'p311', 'p313', 'p314'
]