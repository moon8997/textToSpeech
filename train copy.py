from datasets import load_dataset, DatasetDict, Audio, load_from_disk
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate

import os

# Hugging Face 캐시 경로 변경
os.environ["HF_HOME"] = "D:/huggingface_cache"  # 모델 캐시 경로
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface_cache/datasets"  # 데이터셋 캐시 경로

torch.cuda.memory.set_per_process_memory_fraction(0.9)
torch.cuda.memory.max_split_size_mb = 128

model_id = "openai/whisper-base"
out_dir = 'whisper_test'
epochs = 10
batch_size = 2
# 10 / 32
# 10 / 2
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels

        return batch

def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch['audio']

    batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
    batch['labels'] = tokenizer(batch['text']).input_ids

    return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {'wer': wer}

dataset_cache_dir = "D:/huggingface/datasets/parquet/default-90b83debc148a4d4/0.0.0/9d41700293b5cf3c3cee6167e8c49e37598331b6466506aecb40a8c11b6aa9f6"

if __name__ == "__main__":
    # Parquet 파일에서 데이터셋 로드
    dataset = load_from_disk(dataset_cache_dir)

    # 각 데이터셋을 별도로 추출
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    # # 데이터셋 정보 출력
    # print(train_dataset)
    # print(validation_dataset)

    # Feature Extractor, Tokenizer, Processor 로드
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language='English', task='transcribe')
    processor = WhisperProcessor.from_pretrained(model_id, language='English', task='transcribe')

    # ATC 데이터셋 로드
    # atc_dataset_train = load_dataset('Dbdn/atc-test', split='train')
    # atc_dataset_valid = load_dataset('Dbdn/atc-test', split='validation')

    # print(atc_dataset_train)
    # print(atc_dataset_valid)

    # 오디오 데이터를 Audio 형식으로 캐스트
    atc_dataset_train = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
    atc_dataset_valid = validation_dataset.cast_column('audio', Audio(sampling_rate=16000))

    # print(atc_dataset_train[0])

    atc_dataset_train = atc_dataset_train.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer)
    )
    atc_dataset_valid = atc_dataset_valid.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer)
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load('wer')

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir, 
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, 
        learning_rate=0.00001,
        warmup_steps=1000,
        bf16=False,
        fp16=True,
        num_train_epochs=epochs,
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        generation_max_length=225,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        dataloader_num_workers=8,
        save_total_limit=2,
        lr_scheduler_type='constant',
        seed=42,
        data_seed=42
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=atc_dataset_train,
        eval_dataset=atc_dataset_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    model.save_pretrained(f"{out_dir}/best_model")
    tokenizer.save_pretrained(f"{out_dir}/best_model")
    processor.save_pretrained(f"{out_dir}/best_model")
