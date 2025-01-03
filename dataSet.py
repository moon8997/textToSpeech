from datasets import Dataset, Audio, DatasetDict
import os
from transformers import WhisperProcessor

# 오디오 및 텍스트 파일이 저장된 경로 설정
audio_dir = "dataset copy/audio"
text_dir = "dataset copy/text"

# Whisper Processor 로드
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 파일 목록 가져오기
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
data = {'audio': [], 'text': []}

# 오디오 파일과 해당 텍스트 파일을 연결
for audio_file in audio_files:
    # 오디오 파일 경로 설정
    audio_path = os.path.join(audio_dir, audio_file)
    data['audio'].append(audio_path)
    
    # 해당 텍스트 파일 찾기
    text_file = os.path.splitext(audio_file)[0] + '.txt'
    text_path = os.path.join(text_dir, text_file)
    
    # 텍스트 파일이 존재하는지 확인
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
            data['text'].append(transcript)
    else:
        print(f"Warning: No transcript found for {audio_file}")
        data['text'].append("")

# 데이터셋 생성
dataset = Dataset.from_dict(data)

# 오디오 파일을 Audio 형식으로 변환
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 데이터셋 검증
print(dataset)

# 데이터셋 분리 (예: train, validation)
train_test_split = dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

# Parquet 파일로 저장
train_parquet_path = "dataset/dataset/train.parquet"
validation_parquet_path = "dataset/dataset/validation.parquet"
dataset_dict["train"].to_parquet(train_parquet_path)
dataset_dict["validation"].to_parquet(validation_parquet_path)

print(f"Train dataset saved to {train_parquet_path}")
print(f"Validation dataset saved to {validation_parquet_path}")

# Hugging Face에 업로드 (예시)
dataset_dict.push_to_hub("Dbdn/atc-tst")
