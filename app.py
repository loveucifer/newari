
### still a work in progress ###

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from pathlib import Path

# --- Configuration ---
# This class holds the configuration for loading the model and tokenizer.
# Make sure the paths and language codes match your trained model.
class TranslationConfig:
    """Configuration for the translation model."""
    # Directory where the fine-tuned model and tokenizer are saved.
    model_dir = "./newari_model"
    # Language code for English (as used during training).
    en_code = "eng_Latn"
    # Language code for Newari (as used during training).
    newari_code = "new_Deva"
    # Generation parameters
    max_length = 64
    num_beams = 5  # Using more beams can lead to better quality translations.

# --- Translator Class ---
class Translator:
    """A class to handle loading the model and performing translations."""

    def __init__(self, config: TranslationConfig):
        """
        Initializes the Translator by loading the model and tokenizer.
        
        Args:
            config (TranslationConfig): The configuration object.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        model_path = Path(self.config.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found at '{self.config.model_dir}'. "
                "Please ensure you have trained the model and it's saved in the correct location."
            )

        # Load the fine-tuned model and tokenizer from the specified directory.
        logging.info(f"Loading model from {self.config.model_dir}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)

        # CORRECTED: Manually add the language tokens to the tokenizer's vocabulary if they don't exist.
        # This is the most robust way to ensure the tokenizer recognizes the language codes.
        special_tokens_to_add = []
        if self.config.en_code not in self.tokenizer.get_vocab():
            special_tokens_to_add.append(self.config.en_code)
        if self.config.newari_code not in self.tokenizer.get_vocab():
            special_tokens_to_add.append(self.config.newari_code)

        if special_tokens_to_add:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
            logging.info(f"Manually added special tokens: {special_tokens_to_add}")

        logging.info("Model and tokenizer loaded successfully.")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates a given text from a source language to a target language.

        Args:
            text (str): The input text to translate.
            source_lang (str): The language code of the input text (e.g., "eng_Latn").
            target_lang (str): The language code for the translated output (e.g., "new_Deva").

        Returns:
            str: The translated text.
        """
        if not text.strip():
            return ""

        # Set the source language for the tokenizer. This is crucial for the NLLB model.
        self.tokenizer.src_lang = source_lang

        # 1. Tokenize the input text.
        # The tokenizer converts the text into a format the model can understand (input IDs).
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length).to(self.device)

        # 2. Generate the translation.
        # Use `convert_tokens_to_ids` for a reliable way to get the language token ID.
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)
        
        # Check if the token was found. If not, it's an unknown token.
        if forced_bos_token_id == self.tokenizer.unk_token_id:
             raise ValueError(
                f"Target language code '{target_lang}' not found in tokenizer vocabulary. "
                "Please ensure it matches the code used during training."
             )

        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            early_stopping=True,
        )

        # 3. Decode the generated tokens back into text.
        # We skip special tokens like <pad> or </s>.
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated_text

# --- Main execution block ---
if __name__ == "__main__":
    # Setup basic logging to see the progress.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Create configuration and translator instances.
        config = TranslationConfig()
        translator = Translator(config)

        # --- Example 1: English to Newari ---
        print("\n" + "="*30)
        print(" translating English to Newari")
        print("="*30)
        english_texts = [
            "Hello, how are you?",
            "What is your name?",
            "I am learning the Newari language.",
            "This is a beautiful place."
        ]

        for text in english_texts:
            translated = translator.translate(text, source_lang=config.en_code, target_lang=config.newari_code)
            print(f"  English: {text}")
            print(f"  Newari: {translated}\n")

        # --- Example 2: Newari to English ---
        print("\n" + "="*30)
        print(" translating Newari to English")
        print("="*30)
        # Note: These are example sentences. Replace with actual Newari text.
        newari_texts = [
            "जि छगू नेवाः हुँ ।", # I am a Newar.
            "छिगु नां छु खः ?", # What is your name?
            "नेपालभाषा सिक्दै छु ।" # I am learning the Nepal Bhasa language.
        ]

        for text in newari_texts:
            translated = translator.translate(text, source_lang=config.newari_code, target_lang=config.en_code)
            print(f"  Newari: {text}")
            print(f"  English: {translated}\n")

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
