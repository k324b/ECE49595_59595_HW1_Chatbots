#SETUP INSTRUCTIONS

#in VSCode or terminal:

git clone https://github.com/k324b/ECE495.git

cd ECE495

#if in VSCode, best to file, open folder, open ECE495

#create virtual environment (might take a bit)

python -m venv venv

#activate

#Windows cmd.exe

venv\Scripts\activate.bat

#Windows powershell

venv\Scripts\Activate.ps1

#MacOS

source venv/bin/activate

#Every time you are working with the code, you will need to run the activate command to be in the virtual environment or you will not have your libraries

#Every time you are done with the code, run

deactivate

#This takes you out of the virtual environment

#install libraries in virtual environment

pip install numpy openai-whisper torch SpeechRecognition rich langchain langchain-core langchain-community langchain-ollama pyttsx3 pyaudio

#make sure you have your bot running
#in the code, in main.py, update line llm = OllamaLLM(model="bot"), currently line 17 to llm = OllamaLLM(model="yourmodel")
#you do not need to change the prompt, this does nothing

#finally, run main.py

python main.py or python3 main.py(i think in Mac?)

#should take a while to run the first time
<img width="1003" height="410" alt="image" src="https://github.com/user-attachments/assets/3bb44949-3e02-4efb-9bba-f8345905a234" />

