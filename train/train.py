#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (
    unicode_literals,
    print_function
)

import torch
import evaluate
import logging
import sys
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
    WhisperForConditionalGeneration,
    TrainerCallback
)

# ✅ Set Up Logging to File
log_file = "training_log_run_2.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),  # Append mode to keep logs
        logging.StreamHandler(sys.stdout)  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info("🚀 Starting Whisper Small Fine-Tuning")

# ✅ Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ✅ Load dataset
logger.info("Loading Common Voice Swahili dataset...")
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

logger.info("Removing unwanted columns...")
COLS = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
common_voice = common_voice.remove_columns(COLS)

logger.info(f"First training sample: {common_voice['train'][0]}")

# ✅ Load processors
logger.info("Loading WhisperFeatureExtractor & Tokenizer...")
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

# ✅ Resample input audio from 48kHz to 16kHz
logger.info("Resampling audio to 16kHz...")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# ✅ Data preprocessing function
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

logger.info("Applying data preprocessing...")
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1
)

# ✅ Load Pre-Trained Model
logger.info("Loading Whisper Small model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

# ✅ Set model generation config
model.generation_config.language = "swahili"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

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

logger.info("Initializing Data Collator...")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

# ✅ Load WER Metric for Evaluation
logger.info("Loading Word Error Rate (WER) metric...")
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ✅ Define Training Configuration
logger.info("Defining training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-sw",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,  # Enable Mixed Precision Training
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

# Move model to CUDA again (safety check)
model.to(device)

# ✅ Initialize Trainer
logger.info("Initializing Trainer...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# ✅ Save Processor before Training
logger.info("Saving processor before training...")
processor.save_pretrained(training_args.output_dir)

# ✅ Log Trainer Details Before Training
logger.info("🚀 Starting Training with the following configuration:")
logger.info(f"Trainer Arguments: {training_args}")
logger.info(f"Training Dataset Size: {len(common_voice['train'])}")
logger.info(f"Evaluation Dataset Size: {len(common_voice['test'])}")

# ✅ Custom Callback for Logging Training Progress
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Logs training metrics at each step."""
        if logs:
            logger.info(f"Training Step: {state.global_step}")
            for key, value in logs.items():
                logger.info(f"{key}: {value}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Logs summary at the end of each epoch."""
        logger.info(f"🚀 Epoch {state.epoch} completed.")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Logs evaluation results."""
        logger.info("🔍 Evaluation Metrics:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

    def on_save(self, args, state, control, **kwargs):
        """Logs checkpoint saving."""
        logger.info(f"💾 Model checkpoint saved at step {state.global_step}.")

# ✅ Add Callback to Trainer
trainer.add_callback(LoggingCallback())

# ✅ Start Training with Full Logging
logger.info("🚀 Training started...")

# ✅ Start Training with Exception Handling
logger.info("🚀 Starting training...")
try:
    trainer.train()
    logger.info("🎉 Training Complete!")
except Exception as e:
    logger.error(f"🚨 Training crashed: {e}", exc_info=True)
