#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (
    unicode_literals,
    print_function
)

import torch
import evaluate
from time import strftime
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration
)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

print(strftime('%d %b %Y %H:%M:%S'), "Initialize Training\n")

# Load dataset
common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "sw",
    split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "sw",
    split="test"
)

print(strftime('%H:%M:%S'), "Remove unwanted Columns")
COLS = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
common_voice = common_voice.remove_columns(COLS)

print(strftime('%H:%M:%S'), common_voice['train'][0])

# Load processors
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Swahili",
    task="transcribe"
)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Swahili",
    task="transcribe"
)

# Downsample input audio from 48kHz to 16kHz
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print(strftime('%H:%M:%S'), "Map data prep\n")
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1
)

print(strftime('%H:%M:%S'), "Load Pre-Trained Model\n")

# Load model and move to CUDA
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

# Set model generation config
model.generation_config.language = "swahili"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Process input features (keep on CPU)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Process labels (keep on CPU)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS token is appended in tokenization, remove it
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # ❌ DO NOT MOVE TO GPU HERE (Trainer will handle this)
        return batch

print("\nInitialize Data Collator\n")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

print(strftime('%H:%M:%S'), "Define Training Configuration")

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-sw",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,  # ✅ Enable Mixed Precision Training
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


# Move model to CUDA again (redundant safety check)
model.to(device)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("\nSave Processor before Training\n")
processor.save_pretrained(training_args.output_dir)

print(strftime('%H:%M:%S'), "Start Training\n")
trainer.train()

print(strftime('%d %b %Y %H:%M:%S'), "Training Complete\n")
