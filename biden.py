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
stt = whisper.load_model("tiny.en") # better for computer with lower RAM
tts = TextToSpeechService()
llm = OllamaLLM(model="llama3.2:3b")


# prompt for bot to act like Biden
template = """You are former president Joseph R. Biden. You must stay in character at all times. 
You are emphathetic and are known as a "leader with a heart", but you are also known for commiting verbal gaffes
as a result of your old age. You have served your country as a senator for over 30 years and focus on 
uniting the country. You are characterized as an old-school debator with folksy and personal anecdotes. You are also direct
and showcase a fighter persona. Do not narrate actions/gestures/sounds, only include dialogue. Keep your responses limited to
15 seconds or less. The person you will be having a conversation with is former president Donald J. Trump.
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
    recognizer.pause_threshold = 2.0


    with microphone as source:
        console.print("[cyan]Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        console.print("[bold green]System Online. I am listening!")

    # internally gives Biden a prompt so that he starts the conversation off
    initial_input = "You will be debating Donald Trump. Ask him about his policies."
    result = chain.invoke({"input": initial_input})
    response = result['response']
    console.print(f"[cyan]Assistant: {response}")
    tts.long_form_synthesize(response)
    time.sleep(1.0)  

    # begins back and forth conversations
    while True:
        try:
            with microphone as source:
                console.print("\n[magenta]Listening...")
                audio = recognizer.listen(source, phrase_time_limit=20) # chose phrase limit 20 so it listents to the
                                                                        # entire response and does not prematurely reply

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

            # speaks response out loud
            tts.long_form_synthesize(response)
            
            time.sleep(1.0)

        except Exception as e:
            console.print(f"[red]Error: {e}")
            continue
        except KeyboardInterrupt:
            console.print("[yellow]Shutting down...")
            break

if __name__ == "__main__":
    start_assistant()