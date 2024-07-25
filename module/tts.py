from TTS.api import TTS
import torch

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TTS models
tts_female = TTS("tts_models/en/jenny/jenny").to(device)
tts_male = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)  # 퀄리티가 별론데 속도가 빠름
