import whisper
import spacy
from pathlib import Path
from pydub import AudioSegment


class WhisperNLP:
    """
    - Loads LOCAL Whisper model only
    - Loads spaCy model
    - Transcribes audio
    - Performs NLP analysis
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.nlp = None
        current_file = Path(__file__).resolve()
        self.local_model_path = current_file.parent / "models" / "medium.en.pt"


    def load_models(self):

        if not self.local_model_path.exists():
            raise FileNotFoundError(
                f"Whisper model not found at {self.local_model_path}\n"
                "Download it once and place it inside the models/ directory."
            )

        print(f"Loading Whisper model from local file: {self.local_model_path}")
        self.model = whisper.load_model(str(self.local_model_path))

        # ---- LOAD SPACY ----
        print("Loading spaCy model en_core_web_sm...")
        self.nlp = spacy.load("en_core_web_sm")

        print("Models successfully loaded.")

    def transcribe(self, audio_path):
        result = self.model.transcribe(str(audio_path), fp16=False)
        text = result.get("text", "").strip()

        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000

        return {
            "audio_name": Path(audio_path).name,
            "text": text,
            "duration_seconds": duration_seconds
        }

    def analyze(self, text: str):
        doc = self.nlp(text)

        total_words = sum(1 for t in doc if t.is_alpha)
        total_sentences = len(list(doc.sents))
        avg_words_per_sentence = (
            total_words / total_sentences if total_sentences else 0
        )

        noun_count = sum(1 for t in doc if t.pos_ == "NOUN")
        verb_count = sum(1 for t in doc if t.pos_ == "VERB")
        adj_count  = sum(1 for t in doc if t.pos_ == "ADJ")

        word_length_dist = {l: 0 for l in range(3, 11)}
        for t in doc:
            if t.is_alpha and 3 <= len(t.text) <= 10:
                word_length_dist[len(t.text)] += 1

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
