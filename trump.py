import numpy as np
import whisper
import speech_recognition as sr
import torch
import time
from rich.console import Console
from langchain_ollama import OllamaLLM
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("tiny.en")
tts = TextToSpeechService(model_path="en_US-hfc_male-medium.onnx")
llm = OllamaLLM(model="llama3.2:3b") 

template = """You are Donald J. Trump. You must stay in character at all times. 
Your style is confident, competitive, and uses frequent superlatives like 'tremendous', 'disaster', and 'huge'. 
You focus on your record, the economy, and 'winning'. 
In this debate, you are direct and use short, punchy sentences.
{history}
User: {input}
Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=llm,
)

def start_assistant():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        console.print("[cyan]Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        console.print("[bold green]System Online. I am listening!")

    while True:
        try:
            with microphone as source:
                console.print("\n[magenta]Listening...")
                audio = recognizer.listen(source, phrase_time_limit=20)

            with console.status("Transcribing...", spinner="earth"):
                audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                text = stt.transcribe(audio_np, fp16=torch.cuda.is_available())["text"].strip()
                
                if len(text) < 2: continue
                console.print(f"[yellow]You: {text}")

            with console.status("Thinking...", spinner="earth"):
                result = chain.invoke({"input": text})
                response = result['response']
                console.print(f"[cyan]Assistant: {response}")

            tts.speak(response)
            time.sleep(1.0)

        except Exception as e:
            console.print(f"[red]Error: {e}")
            continue
        except KeyboardInterrupt:
            console.print("[yellow]Shutting down...")
            break

if __name__ == "__main__":
    start_assistant()