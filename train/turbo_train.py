#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 🚀 Check GPU Availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 📌 Load Common Voice Swahili Dataset
print("Loading Common Voice Swahili dataset...")
common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "sw", split="train+validation")
test_set = load_dataset("mozilla-foundation/common_voice_11_0", "sw", split="test")

# Remove unwanted columns
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
test_set = test_set.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# 🚀 Load Whisper Turbo Model & Processor
print("Loading Whisper Turbo model...")
model_name = "openai/whisper-large-v3-turbo"  # Turbo version
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Force model to generate Swahili
model.generation_config.language = "swahili"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# 📌 Resample Audio to 16kHz
print("Resampling audio to 16kHz...")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
test_set = test_set.cast_column("audio", Audio(sampling_rate=16000))

# 🚀 Data Preprocessing
def prepare_dataset(batch):
    """Convert audio to log-Mel spectrogram & tokenize text"""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("Processing dataset...")
common_voice = common_voice.map(prepare_dataset, remove_columns=["audio"])
test_set = test_set.map(prepare_dataset, remove_columns=["audio"])

# 🚀 Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

# 🚀 WER Metric for Evaluation
print("Loading Word Error Rate (WER) metric...")
metric = evaluate.load("wer")

def compute_metrics(pred):
    """Calculate Word Error Rate (WER)"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 🚀 Training Configuration (Optimized for 6GB VRAM)
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-turbo-sw",
    per_device_train_batch_size=2,  # 🔹 Lower batch size to fit in 6GB VRAM
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # 🔹 Simulate batch size of 16
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,  # 🔹 Saves memory
    fp16=True,  # 🔹 Mixed Precision
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# 🚀 Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice,
    eval_dataset=test_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Save processor before training
processor.save_pretrained(training_args.output_dir)

# 🚀 Start Fine-Tuning
print("🚀 Starting fine-tuning...")
trainer.train()

print("🎉 Fine-tuning complete!")
