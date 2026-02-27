import io
import wave
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

class TextToSpeechService:
    def __init__(self, model_path="en_US-hfc_male-medium.onnx"):
        self.voice = PiperVoice.load(model_path)
        self.sample_rate = self.voice.config.sample_rate

    def speak(self, text):
        if not text.strip():
            return
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        buf.seek(44)  
        raw_pcm = buf.read()
        audio_array = np.frombuffer(raw_pcm, dtype=np.int16)
        with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
            stream.write(audio_array)