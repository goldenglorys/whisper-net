# Whisper-Net

This project is a voice and text assistant that allows users to ask questions by voice or text and get responses. The assistant uses Whisper for speech-to-text, a language model (llama) for generating responses, and Parler-TTS for text-to-speech.

## Features

- Voice input: Record your voice and get a transcription and response.
- Text input: Type your question and get a response.
- Text-to-speech: Convert the generated response to speech using Parler-TTS.

## Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (venv)

### Installation

1. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   source ./venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install required packages:**

   ```bash
   pip install gradio
   pip install langchain
   pip install langchain-community
   pip install git+https://github.com/openai/whisper.git
   pip install git+https://github.com/huggingface/parler-tts.git
   ```

### Ollama

1. **Download Ollama:**

   - Visit the [Ollama download page](https://ollama.com/download/) and follow the instructions to download and install Ollama.

2. **Install langchain and langchain-community:**

   ```bash
   pip install langchain
   pip install langchain-community
   ```

3. **Download a model (e.g., Llama):**

   ```bash
   ollama run llama3:8b
   ```

   - To check which models are available, run:

     ```bash
     ollama list
     ```

   - For more information on available models, visit the [Ollama library](https://ollama.com/library).

### Gradio

1. **Install Gradio:**

   ```bash
   pip install gradio
   ```

### Whisper

1. **Install Whisper from GitHub:**

   ```bash
   pip install git+https://github.com/openai/whisper.git
   ```

## Running the Application

1. **Run the script:**

   ```bash
   python main.py
   ```

2. **Open the provided URL in your web browser to access the Gradio app.**


## Usage

1. **Record your voice using the microphone or type a question in the text box.**
2. **Click the respective buttons to see the transcription, response, and hear the audio response displayed in the app.**

## Acknowledgments

- **Whisper:** [Whisper GitHub Repository](https://github.com/openai/whisper)
- **Parler-TTS:** [Parler-TTS GitHub Repository](https://github.com/huggingface/parler-tts)
- **Ollama:** [Ollama Documentation](https://ollama.ai/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.