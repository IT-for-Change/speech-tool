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
        Performs simple NLP analysis using spaCy:
        - word count (alphabetic tokens only)
        - sentence count
        """
        doc = self.nlp(text)
        return {
            "word_count": sum(1 for t in doc if t.is_alpha),
            "sentence_count": len(list(doc.sents))
        }

