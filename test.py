from datasets import load_dataset, DatasetDict, Audio

dataset = load_dataset("parquet", data_files={"train": "dataset/dataset/train.parquet", "validation": "dataset/dataset/validation.parquet"})

# 각 데이터셋을 별도로 추출
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

# ATC 데이터셋 로드
atc_dataset_train = load_dataset('Dbdn/atcs', split='train')
atc_dataset_valid = load_dataset('Dbdn/atcs', split='validation')

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

# 데이터셋 정보 출력
print(train_dataset)
print(validation_dataset)

print(atc_dataset_train)
print(atc_dataset_valid)
