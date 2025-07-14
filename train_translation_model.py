import json
import torch
import os
import gc
import numpy as np
import unicodedata
import warnings
import logging
import random
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    EarlyStoppingCallback, DataCollatorForSeq2Seq, TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import sacrebleu
import torch.nn.functional as F
from torch.optim import AdamW

warnings.filterwarnings("ignore")

@dataclass
class OptimizedConfig:
    # Model settings
    model_name: str = "facebook/nllb-200-distilled-600M"
    en_code: str = "eng_Latn"
    newari_code: str = "new_Deva"
    
    # Data settings
    data_file: str = "translations.json"
    max_length: int = 64  # Reduced for speed
    
    # Training settings
    output_dir: str = "./newari_model"
    num_epochs: int = 15  # Reduced - early stopping will handle this
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    
    # Batch settings
    train_batch_size: int = 4
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"
    ])
    
    # Generation settings
    num_beams: int = 1  # Greedy for speed
    label_smoothing: float = 0.15
    dropout_rate: float = 0.15
    
    # Evaluation settings
    eval_steps: int = 200
    save_steps: int = 400
    eval_accumulation_steps: int = 4

def setup_logging(config: OptimizedConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger(__name__)

def setup_device():
    if torch.backends.mps.is_available():
        # Set memory fraction for MPS
        torch.mps.set_per_process_memory_fraction(0.8)
        device = torch.device("mps")
        torch.mps.empty_cache()
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        return device
    return torch.device("cpu")

def cleanup_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

class EnhancedMonitoringCallback(TrainerCallback):
    def __init__(self, output_dir: str, patience: int = 3, min_delta: float = 0.5):
        self.output_dir = Path(output_dir)
        self.patience = patience
        self.min_delta = min_delta
        
        # Tracking variables
        self.best_metric = -float('inf')
        self.best_step = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_chrF': [],
            'eval_BLEU': [],
            'learning_rate': [],
            'steps': [],
            'epochs': []
        }
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            epoch = state.epoch
            
            # Training metrics
            if 'loss' in logs:
                self.history['train_loss'].append(logs['loss'])
                self.history['steps'].append(step)
                self.history['epochs'].append(epoch)
                
            if 'learning_rate' in logs:
                self.history['learning_rate'].append(logs['learning_rate'])
                
            # Evaluation metrics
            if 'eval_loss' in logs:
                self.history['eval_loss'].append(logs['eval_loss'])
                
            if 'eval_chrF' in logs:
                current_metric = logs['eval_chrF']
                self.history['eval_chrF'].append(current_metric)
                
                # Check for improvement
                if current_metric > self.best_metric + self.min_delta:
                    self.best_metric = current_metric
                    self.best_step = step
                    self.patience_counter = 0
                    print(f"✓ New best chrF: {current_metric:.4f} at step {step}")
                else:
                    self.patience_counter += 1
                    print(f"⚠ No improvement for {self.patience_counter} evaluations")
                
                # Early stopping check
                if self.patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered after {self.patience_counter} evaluations without improvement")
                    print(f"   Best chrF: {self.best_metric:.4f} at step {self.best_step}")
                    control.should_training_stop = True
                    
            if 'eval_BLEU' in logs:
                self.history['eval_BLEU'].append(logs['eval_BLEU'])
                
            # Save history after each log
            self.save_history()
            
    def save_history(self):
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def plot_metrics(self):
        if not self.history['train_loss']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training/Validation Loss
        ax1 = axes[0, 0]
        if self.history['train_loss']:
            ax1.plot(self.history['steps'], self.history['train_loss'], 
                    label='Train Loss', color='blue', alpha=0.7)
        if self.history['eval_loss']:
            eval_steps = self.history['steps'][::max(1, len(self.history['steps'])//len(self.history['eval_loss']))][:len(self.history['eval_loss'])]
            ax1.plot(eval_steps, self.history['eval_loss'], 
                    label='Eval Loss', color='red', alpha=0.7, marker='o')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # chrF Score
        ax2 = axes[0, 1]
        if self.history['eval_chrF']:
            eval_steps = self.history['steps'][::max(1, len(self.history['steps'])//len(self.history['eval_chrF']))][:len(self.history['eval_chrF'])]
            ax2.plot(eval_steps, self.history['eval_chrF'], 
                    label='chrF Score', color='green', marker='o')
            ax2.axhline(y=42, color='red', linestyle='--', alpha=0.7, label='Target (42)')
            ax2.axhline(y=self.best_metric, color='orange', linestyle='--', alpha=0.7, label=f'Best ({self.best_metric:.2f})')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('chrF Score')
        ax2.set_title('chrF Score Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # BLEU Score
        ax3 = axes[1, 0]
        if self.history['eval_BLEU']:
            eval_steps = self.history['steps'][::max(1, len(self.history['steps'])//len(self.history['eval_BLEU']))][:len(self.history['eval_BLEU'])]
            ax3.plot(eval_steps, self.history['eval_BLEU'], 
                    label='BLEU Score', color='purple', marker='o')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('BLEU Score')
        ax3.set_title('BLEU Score Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate
        ax4 = axes[1, 1]
        if self.history['learning_rate']:
            ax4.plot(self.history['steps'], self.history['learning_rate'], 
                    label='Learning Rate', color='brown')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def on_train_end(self, args, state, control, **kwargs):
        self.plot_metrics()
        
        # Save final summary
        summary = {
            'best_chrF': self.best_metric,
            'best_step': self.best_step,
            'total_steps': state.global_step,
            'total_epochs': state.epoch,
            'target_achieved': self.best_metric >= 42,
            'early_stopped': self.patience_counter >= self.patience
        }
        
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n{'='*50}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Best chrF Score: {self.best_metric:.4f}")
        print(f"Target (42) {'✓ ACHIEVED' if summary['target_achieved'] else '✗ NOT ACHIEVED'}")
        print(f"Early Stopped: {'Yes' if summary['early_stopped'] else 'No'}")
        print(f"Total Steps: {state.global_step}")
        print(f"Total Epochs: {state.epoch:.2f}")

class OptimizedDataProcessor:
    def __init__(self, config: OptimizedConfig, logger):
        self.config = config
        self.logger = logger

    def normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize('NFC', text.strip())
        return ' '.join(text.split())

    def inject_noise(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        
        # Only one type of noise per text
        if random.random() < 0.1:  # Word duplication
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, words[idx])
        elif random.random() < 0.05 and len(words) > 2:  # Word deletion
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        
        return ' '.join(words)

    def create_synthetic_pairs(self, en_text: str, new_text: str) -> List[Dict]:
        pairs = [
            {"source_lang": self.config.en_code, "target_lang": self.config.newari_code, 
             "source_text": en_text, "target_text": new_text},
            {"source_lang": self.config.newari_code, "target_lang": self.config.en_code, 
             "source_text": new_text, "target_text": en_text}
        ]
        
        # Only add noise 10% of the time
        if random.random() < 0.1:
            noisy_en = self.inject_noise(en_text)
            pairs.append({
                "source_lang": self.config.en_code, 
                "target_lang": self.config.newari_code,
                "source_text": noisy_en, 
                "target_text": new_text
            })
        
        return pairs

    def load_and_prepare_data(self) -> Dataset:
        self.logger.info(f"Loading data from {self.config.data_file}")
        with open(self.config.data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        processed_data = []
        for item in raw_data:
            en = self.normalize_text(item.get('en', ''))
            new = self.normalize_text(item.get('new', ''))
            
            if len(en.split()) >= 2 and len(new.split()) >= 2:
                len_ratio = len(en) / len(new) if new else 0
                if 0.2 < len_ratio < 5.0:
                    processed_data.extend(self.create_synthetic_pairs(en, new))

        self.logger.info(f"Generated {len(processed_data)} training examples from {len(raw_data)} pairs")
        return Dataset.from_list(processed_data)

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.label_smoothing = kwargs.pop('label_smoothing', 0.15)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, *args, **kwargs):
        cleanup_memory()
        return super().evaluation_loop(*args, **kwargs)

class OptimizedTranslationPipeline:
    def __init__(self, config: OptimizedConfig, device: torch.device, logger):
        self.config = config
        self.device = device
        self.logger = logger
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()

    def _setup_tokenizer(self):
        self.logger.info(f"Loading tokenizer for {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.src_lang = self.config.en_code
        tokenizer.tgt_lang = self.config.newari_code
        return tokenizer

    def _setup_model(self):
        self.logger.info("Loading and configuring model")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            use_cache=False
        )
        
        # FIXED: Apply dropout rate to all layers correctly
        def apply_dropout_recursively(module, dropout_rate):
            """Recursively apply dropout rate to all dropout layers"""
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Dropout):
                    # Set the dropout probability (p) to the desired rate
                    child.p = dropout_rate
                else:
                    # Recursively apply to children
                    apply_dropout_recursively(child, dropout_rate)
        
        # Apply dropout to the entire model
        apply_dropout_recursively(model, self.config.dropout_rate)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False
        )
        
        model = get_peft_model(model, lora_config)
        model.to(self.device)
        model.print_trainable_parameters()
        return model

    def preprocess_function(self, examples):
        self.tokenizer.src_lang = examples['source_lang'][0]
        model_inputs = self.tokenizer(
            examples["source_text"],
            max_length=self.config.max_length,
            truncation=True,
            padding=False
        )
        
        with self.tokenizer.as_target_tokenizer():
            self.tokenizer.tgt_lang = examples['target_lang'][0]
            labels = self.tokenizer(
                examples["target_text"],
                max_length=self.config.max_length,
                truncation=True,
                padding=False
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        try:
            chrf_score = sacrebleu.corpus_chrf(decoded_preds, decoded_labels, word_order=2).score
            bleu_score = sacrebleu.corpus_bleu(decoded_preds, decoded_labels).score
        except:
            chrf_score, bleu_score = 0.0, 0.0
        
        return {"chrF": chrf_score, "BLEU": bleu_score}

    def get_training_args(self):
        return Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Learning rate settings
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine_with_restarts",
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="chrF",
            greater_is_better=True,
            
            # CRITICAL FIX: Disable multiprocessing for macOS MPS
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Changed from 2 to 0
            
            # Memory optimizations
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            # Generation settings
            predict_with_generate=True,
            generation_max_length=self.config.max_length,
            generation_num_beams=self.config.num_beams,
            
            # Regularization
            label_smoothing_factor=self.config.label_smoothing,
            
            # Logging
            logging_steps=50,
            report_to="none",
            
            # Misc
            remove_unused_columns=False,
            dataloader_drop_last=True,
            eval_accumulation_steps=self.config.eval_accumulation_steps,
            
            # Optimizer settings
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-6,
            max_grad_norm=0.5,
            
            # Mixed precision - disabled for MPS stability
            fp16=False,
            bf16=False,
        )

    def train(self, train_dataset, eval_dataset):
        self.logger.info("Starting optimized training")
        
        training_args = self.get_training_args()
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
            max_length=self.config.max_length,
        )

        # Initialize monitoring
        monitor = EnhancedMonitoringCallback(
            output_dir=self.config.output_dir,
            patience=3,
            min_delta=0.5
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                monitor,
                EarlyStoppingCallback(early_stopping_patience=2)
            ],
            label_smoothing=self.config.label_smoothing,
        )

        # Start training
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        self.logger.info(f"Training completed in {(end_time - start_time)/3600:.2f} hours")
        
        # Cleanup
        del trainer
        cleanup_memory()
        
        return monitor

def main():
    # Set multiprocessing start method for macOS compatibility
    if hasattr(torch.multiprocessing, 'set_start_method'):
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # Configuration
    config = OptimizedConfig()
    logger = setup_logging(config)
    device = setup_device()
    
    logger.info(f"Starting optimized training on {device}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Max length: {config.max_length}")
    logger.info(f"Batch size: {config.train_batch_size} x {config.gradient_accumulation_steps} = {config.train_batch_size * config.gradient_accumulation_steps}")
    
    # Initial cleanup
    cleanup_memory()
    
    # Load and prepare data
    data_processor = OptimizedDataProcessor(config, logger)
    dataset = data_processor.load_and_prepare_data()
    
    # Split data
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.125, seed=42)
    
    train_dataset = train_val_split['train']
    eval_dataset = train_val_split['test']
    test_dataset = train_test_split['test']
    
    logger.info(f"Dataset splits - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize pipeline
    pipeline = OptimizedTranslationPipeline(config, device, logger)
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        pipeline.preprocess_function,
        batched=True,
        batch_size=config.train_batch_size,
        remove_columns=dataset.column_names,
        num_proc=1  # Force single process for tokenization
    )
    
    eval_tokenized = eval_dataset.map(
        pipeline.preprocess_function,
        batched=True,
        batch_size=config.eval_batch_size,
        remove_columns=dataset.column_names,
        num_proc=1  # Force single process for tokenization
    )
    
    # Memory cleanup before training
    cleanup_memory()
    
    # Train the model
    monitor = pipeline.train(train_tokenized, eval_tokenized)
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    
    # Clean up pipeline
    del pipeline
    cleanup_memory()
    
    # Load best model for testing
    test_pipeline = OptimizedTranslationPipeline(config, device, logger)
    
    test_tokenized = test_dataset.map(
        test_pipeline.preprocess_function,
        batched=True,
        batch_size=config.eval_batch_size,
        remove_columns=dataset.column_names,
        num_proc=1  # Force single process for tokenization
    )
    
    test_trainer = Seq2SeqTrainer(
        model=test_pipeline.model,
        args=Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            predict_with_generate=True,
            generation_max_length=config.max_length,
            generation_num_beams=config.num_beams,
            per_device_eval_batch_size=config.eval_batch_size,
            report_to="none",
            fp16=False,
            dataloader_num_workers=0,  # Disable multiprocessing
            dataloader_pin_memory=False,
        ),
        tokenizer=test_pipeline.tokenizer,
        compute_metrics=test_pipeline.compute_metrics,
    )
    
    test_results = test_trainer.evaluate(test_tokenized)
    final_chrf = test_results.get('eval_chrF', 0)
    final_bleu = test_results.get('eval_BLEU', 0)
    
    logger.info(f"FINAL TEST RESULTS:")
    logger.info(f"chrF: {final_chrf:.4f}")
    logger.info(f"BLEU: {final_bleu:.4f}")
    
    # Save test results
    with open(f"{config.output_dir}/final_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Final summary
    if final_chrf >= 42:
        logger.info(f"🎉 SUCCESS: Target chrF ≥42 achieved! ({final_chrf:.4f})")
    else:
        logger.info(f"📊 RESULT: chrF={final_chrf:.4f} (target: 42)")
    
    if monitor.best_metric >= 42:
        logger.info(f"🎯 Training target achieved: {monitor.best_metric:.4f}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()