import whisper
import spacy
from pathlib import Path
from pydub import AudioSegment


class WhisperNLP:
    """
    A helper class that handles:
    - Loading local or online Whisper speech-to-text model
    - Loading the spaCy language model
    - Transcribing audio files
    - Performing basic NLP analysis on transcribed text
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.nlp = None

        # Default local model path
        current_file = Path(__file__).resolve()
        self.local_model_path = current_file.parent / "models" / "large-v3-turbo.pt"

    def load_models(self):

        # ---- LOAD WHISPER ----
        if self.local_model_path.exists():
            print(f"Loading Whisper model from local file: {self.local_model_path}")
            self.model = whisper.load_model(str(self.local_model_path))
        else:
            print("Local Whisper model not found. Downloading from internet...")
            self.model = whisper.load_model(self.config.model_name)

        # ---- LOAD SPACY ----
        print("Loading spaCy model en_core_web_sm...")
        self.nlp = spacy.load("en_core_web_sm")

        print("Models successfully loaded.")

    def transcribe(self, audio_path):
        """
        Transcribes audio using the loaded Whisper model and returns:
        - audio file name
        - transcribed text
        - duration of audio in seconds
        """

        result = self.model.transcribe(str(audio_path), fp16=False)
        text = result.get("text", "").strip()

        audio_name = Path(audio_path).name

        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000
        
        return {
            "audio_name": audio_name,
            "text": text,
            "duration_seconds": duration_seconds
        }

    def analyze(self, text: str):
        """
        Performs NLP analysis using spaCy:
        1. Story length (total word count)
        2. Total sentence count
        3. Average number of words per sentence
        4. POS categorization – nouns, verbs, adjectives
        5. Word length distribution for words of length 3 to 10 letters
        """

        doc = self.nlp(text)

        # 1. total word count 
        total_words = sum(1 for token in doc if token.is_alpha)

        # 2. total sentence count
        total_sentences = len(list(doc.sents))

        # 3. average words per sentence
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

        # 4. POS counts 
        noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        adj_count  = sum(1 for token in doc if token.pos_ == "ADJ")

        # 5. word length distribution for 3–10 letter words
        word_length_dist = {length: 0 for length in range(3, 11)}
        for token in doc:
            if token.is_alpha:
                l = len(token.text)
                if 3 <= l <= 10:
                    word_length_dist[l] += 1

        return {
            "word_count": total_words,
            "sentence_count": total_sentences,
            "avg_words_per_sentence": avg_words_per_sentence,
            "pos_counts": {
                "nouns": noun_count,
                "verbs": verb_count,
                "adjectives": adj_count
            },
            "word_length_distribution": word_length_dist
        }


