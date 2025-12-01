from pathlib import Path

class AudioNLPConfig:
    def __init__(self):
        self.audio_dir = Path("audio_inputs")
        self.transcript_dir = Path("audio_transcripts")
        self.model_name = "small"
        self.chunk_size = 16384


class AudioManager:

    def __init__(self, config: AudioNLPConfig):
        self.config = config

        # Ensure directories exist
        self.config.audio_dir.mkdir(exist_ok=True)
        self.config.transcript_dir.mkdir(exist_ok=True)

