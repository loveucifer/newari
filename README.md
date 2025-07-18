# English-Newari and vice versa  Translation Model

This project provides scripts to fine-tune and use a machine translation model for translating between English (`eng_Latn`) and Newari (`new_Deva`) using the NLLB-200-distilled-600M model from Hugging Face, optimized with PyTorch and LoRA (Low-Rank Adaptation) for efficient training.

## Overview

The project consists of two main scripts:
- **`train_translation_model.py`**: Fine-tunes the NLLB-200 model on a parallel English-Newari dataset, optimizing for chrF and BLEU metrics.
- **`test_translation_model.py`**: Loads the fine-tuned model to perform bidirectional translations (English ↔ Newari or vice versa).

The model is designed to run on CPU, GPU (CUDA), or MPS (Apple Silicon), with memory-efficient configurations for training and inference.

## Features
- Fine-tunes the NLLB-200-distilled-600M model using LoRA for parameter-efficient training.
- Supports bidirectional translation (English to Newari and Newari to English).
- Includes data preprocessing with text normalization and synthetic pair generation.
- Monitors training with chrF and BLEU metrics, saving plots and logs.
- Handles resource cleanup for memory efficiency.

## Requirements
- Python 3.8+
- PyTorch (compatible with CUDA or MPS if available)
- Hugging Face Transformers
- Datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- SacreBLEU
- Matplotlib
- NumPy

Install dependencies using:
```bash
pip install torch transformers datasets peft sacrebleu matplotlib numpy
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Prepare the Dataset**:
   - Create a `translations.json` file with parallel English-Newari sentences. Example format:
     ```json
     [
         {"en": "Hello, how are you?", "new": "नमस्ते, छु हाल छ??"},
         {"en": "What is your name?", "new": "छिगु नां छु खः ?"}
     ]
     ```
   - Place `translations.json` in the project root.

3. **Ensure Hardware Compatibility**:
   - The scripts automatically detect and use CUDA, MPS (Apple Silicon), or CPU.
   - For MPS, ensure PyTorch is configured for Apple Silicon.

## Usage

### 1. Training the Model
Run `train_translation_model.py` to fine-tune the model:
```bash
python train_translation_model.py
```
- **Input**: `translations.json` (parallel dataset).
- **Output**: Fine-tuned model and tokenizer saved in `./newari_model`.
- **Logs and Metrics**:
  - Training history: `./newari_model/training_history.json`
  - Metrics plot: `./newari_model/training_metrics.png`
  - Final test results: `./newari_model/final_test_results.json`
- **Key Configurations** (edit in `train_translation_model.py`):
  - `model_name`: Base model (default: `facebook/nllb-200-distilled-600M`).
  - `num_epochs`: Number of training epochs (default: 15).
  - `learning_rate`: Learning rate (default: 5e-5).
  - `max_length`: Maximum sequence length (default: 64).
  - `lora_r`, `lora_alpha`, `lora_dropout`: LoRA parameters.

The script splits the dataset into training, validation, and test sets, applies data augmentation (e.g., word duplication), and uses early stopping based on chrF score (target: ≥42).

### 2. Translating Text
Run `test_translation_model.py` to perform translations using the fine-tuned model:
```bash
python test_translation_model.py
```
- **Requirements**: The `./newari_model` directory must exist (generated by training).
- **Output**: Example translations for predefined English and Newari sentences, printed to the console.
- **Example Output**:
  ```
  ==============================
   translating English to Newari
  ==============================
    English: Hello, how are you?
    Newari: नमस्ते, छु हाल छ??

    English: What is your name?
    Newari: छिगु नां छु खः ?

  ==============================
   translating Newari to English
  ==============================
    Newari: जि छगू नेवाः हुँ ।
    English: I am a Newar.
  ```

To integrate translation into your code:
```python
from test_translation_model import Translator, TranslationConfig

config = TranslationConfig()
translator = Translator(config)
# English to Newari
result = translator.translate("I am learning the Newari language.", "eng_Latn", "new_Deva")
print(result)
# Newari to English
result = translator.translate("नेपालभाषा सिक्दै छु ।", "new_Deva", "eng_Latn")
print(result)
```

### Configuration
Key settings in `test_translation_model.py`:
- `model_dir`: Path to the fine-tuned model (default: `./newari_model`).
- `en_code`, `newari_code`: Language codes (default: `eng_Latn`, `new_Deva`).
- `max_length`: Maximum sequence length for translation (default: 64).
- `num_beams`: Number of beams for beam search (default: 5).

## Troubleshooting
- **FileNotFoundError**: Ensure `./newari_model` exists (run `train_translation_model.py` first) and `translations.json` is in the root directory.
- **Poor Translation Quality**: Check the quality and size of `translations.json`. Increase `num_epochs` or adjust `learning_rate` in `train_translation_model.py`.
- **Memory Issues**: Reduce `train_batch_size` or `gradient_accumulation_steps` in `train_translation_model.py`.
- **Language Code Errors**: Verify that `eng_Latn` and `new_Deva` match the tokenizer’s vocabulary.

## Optimization Notes
- **LoRA**: Reduces memory usage by fine-tuning only a subset of parameters. Adjust `lora_r` and `lora_alpha` for trade-offs between efficiency and performance.
- **Dropout**: Applied uniformly across layers (`dropout_rate=0.15`) to prevent overfitting.
- **Label Smoothing**: Set to 0.15 to improve generalization.
- **Gradient Checkpointing**: Enabled to reduce memory usage during training.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions or issues, please open an issue on the repository or contact the maintainer.


## Dataset
dataset is soemthing i have built with my own research containing nearly 5000 high quality texts that are purpose built and will be uploaded soon
