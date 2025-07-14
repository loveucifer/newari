import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from pathlib import Path
import json
import datetime

# --- Configuration ---
class AnalysisConfig:
    """Configuration for the translation analysis."""
    model_dir = "./newari_model"
    en_code = "eng_Latn"
    newari_code = "new_Deva"
    # Generation parameters
    max_length = 64
    # Use beam search to find better candidates
    num_beams = 5
    # The number of translation candidates to generate for each input sentence
    num_return_sequences = 5
    # Output file for the analysis
    output_file = "translation_analysis.json"

# --- Translator Class for Analysis ---
class Translator:
    """A class to handle loading the model and generating multiple translation candidates."""
    def __init__(self, config: AnalysisConfig):
        """Initializes the Translator by loading the model and tokenizer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        model_path = Path(self.config.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at '{self.config.model_dir}'.")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)

        # Manually add language tokens if they don't exist
        special_tokens_to_add = [self.config.en_code, self.config.newari_code]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_translation_candidates(self, text: str, source_lang: str, target_lang: str) -> list[str]:
        """
        Translates a text and returns a list of candidate translations.
        """
        if not text.strip():
            return []

        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length).to(self.device)
        
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)
        if forced_bos_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Target language code '{target_lang}' not found in tokenizer's vocabulary.")

        # Generate multiple sequences. This is the key change for analysis.
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            num_return_sequences=self.config.num_return_sequences,
            early_stopping=True,
        )
        
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# --- Main execution block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        config = AnalysisConfig()
        translator = Translator(config)
        
        # --- Texts to Analyze ---
        # You can add any sentences you want to test here.
        english_texts_to_analyze = [
            "Hello, how are you?",
            "What is your name?",
            "I am learning the Newari language.",
            "This is a beautiful place.",
            "The sun is shining today."
        ]
        newari_texts_to_analyze = [
            "जि छगू नेवाः हुँ ।",
            "छिगु नां छु खः ?",
            "नेपालभाषा सिक्दै छु ।",
            "थन तसकं बांलाः ।"
        ]
        
        analysis_results = []

        logging.info("--- Analyzing English to Newari Translations ---")
        for text in english_texts_to_analyze:
            candidates = translator.get_translation_candidates(text, source_lang=config.en_code, target_lang=config.newari_code)
            
            structured_candidates = [
                {"rank": i + 1, "text": cand_text, "human_rating (1-5)": None, "is_correct (true/false)": None}
                for i, cand_text in enumerate(candidates)
            ]

            analysis_results.append({
                "source_language": "English",
                "target_language": "Newari",
                "source_text": text,
                "best_candidate": candidates[0] if candidates else None,
                "all_candidates": structured_candidates
            })
            print(f"Processed: '{text}'")

        logging.info("--- Analyzing Newari to English Translations ---")
        for text in newari_texts_to_analyze:
            candidates = translator.get_translation_candidates(text, source_lang=config.newari_code, target_lang=config.en_code)
            
            structured_candidates = [
                {"rank": i + 1, "text": cand_text, "human_rating (1-5)": None, "is_correct (true/false)": None}
                for i, cand_text in enumerate(candidates)
            ]

            analysis_results.append({
                "source_language": "Newari",
                "target_language": "English",
                "source_text": text,
                "best_candidate": candidates[0] if candidates else None,
                "all_candidates": structured_candidates
            })
            print(f"Processed: '{text}'")

        # --- Save results to JSON file ---
        output_data = {
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "model_directory": config.model_dir,
            "generation_parameters": {
                "num_beams": config.num_beams,
                "num_return_sequences": config.num_return_sequences
            },
            "results": analysis_results
        }

        with open(config.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logging.info(f" Analysis complete. Results saved to '{config.output_file}'")

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}")