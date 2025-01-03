from datasets import Dataset, Audio, DatasetDict
import os
from tqdm import tqdm  # tqdm 라이브러리 임포트

# 데이터 경로 설정
base_dir = r"F:\TS_공공데이터 기업 매칭 사업"
audio_base_dir = os.path.join(base_dir, "음성녹음")
text_base_dir = os.path.join(base_dir, "전사작업")

# 데이터셋 초기화
data = {}

# 모든 오디오 파일 수 계산
total_files = sum([len(files) for _, _, files in os.walk(audio_base_dir) if any(f.endswith('.wav') for f in files)])

# 오디오 파일 탐색 (tqdm으로 진행 상황 표시)
index = 1  # 데이터 인덱스
with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
    for root, dirs, files in os.walk(audio_base_dir):
        for file in files:
            if file.endswith('.wav'):
                # 오디오 파일 경로
                audio_path = os.path.join(root, file)

                # 텍스트 파일 경로 추론
                relative_path = os.path.relpath(root, audio_base_dir)  # 오디오 폴더 기준 상대 경로
                text_folder = os.path.join(text_base_dir, relative_path)  # 전사작업 폴더로 변경
                text_file = os.path.splitext(file)[0] + '.txt'
                text_path = os.path.join(text_folder, text_file)

                # 텍스트 파일 확인
                if os.path.exists(text_path):
                    # 텍스트 파일이 있는 경우 데이터셋에 추가
                    with open(text_path, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                        data[index] = {'audio': audio_path, 'text': transcript}  # 인덱스 기반 저장
                        index += 1  # 인덱스 증가
                else:
                    # 텍스트 파일이 없으면 경고 출력
                    print(f"Skipping: No transcript found for {file}. Expected at: {text_path}")
                
                # print(data)
                # tqdm 게이지 업데이트
                pbar.update(1)

# 데이터셋 검증
print(f"Total valid data entries: {len(data)}")
print("Example data entry:", list(data.items())[0])  # 첫 번째 데이터 출력


# 데이터셋 생성
if data:
    # 딕셔너리를 리스트로 변환
    transformed_data = {
        'audio': [entry['audio'] for entry in data.values()],
        'text': [entry['text'] for entry in data.values()]
    }

    # Hugging Face Dataset 생성
    dataset = Dataset.from_dict(transformed_data)

    # 오디오 파일을 Audio 형식으로 변환
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 데이터셋 분리 (train, validation)
    train_test_split = dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })

    # Parquet 파일로 저장
    train_parquet_path = os.path.join(base_dir, "train.parquet")
    validation_parquet_path = os.path.join(base_dir, "validation.parquet")
    dataset_dict["train"].to_parquet(train_parquet_path)
    dataset_dict["validation"].to_parquet(validation_parquet_path)

    print(f"Train dataset saved to {train_parquet_path}")
    print(f"Validation dataset saved to {validation_parquet_path}")
else:
    print("No valid data found. Dataset creation skipped.")


# Hugging Face에 업로드 (예시)
# dataset_dict.push_to_hub("Dbdn/atc-test")
