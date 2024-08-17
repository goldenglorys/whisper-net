import gradio as gr
import whisper
import torch
from langchain_community.llms import Ollama
import json

# Function to load the language spoken in a country
def get_language(country, file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data[country]

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file, sr=16000)
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    result = model.transcribe(audio_tensor, fp16=False)['text']
    return result

# Function to generate responses using the LLM
def generate_response(question, country):
    llm = Ollama(model="llama3:8b")
    lang = get_language(country, file="utils/country_to_language.json")
    context = f"You are a helpful assistant. You answer concisely and only in {lang}."
    query = f"{question}"
    response_lang = llm.invoke(context + query)
    response_english = llm.invoke(query)
    return response_lang, response_english

# Function to handle voice input
def handle_voice_input(audio_file, country):
    transcription = transcribe_audio(audio_file)
    response_lang, response_english = generate_response(transcription, country)
    return transcription, response_lang, response_english

# Function to handle text input
def handle_text_input(question, country):
    response_lang, response_english = generate_response(question, country)
    return response_lang, response_english

# Function to launch the Gradio application
def main():
    # Create a Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice and Text Assistant")
        gr.Markdown("### Ask a question by voice or text and get responses in English and the selected language.")

        # Create a row with audio input and transcription output
        with gr.Row():
            audio = gr.Audio(sources=["microphone"], label="Record your voice", type="filepath", max_length=10)
            transcription = gr.Textbox(label="Transcription")
            response_lang = gr.Textbox(label="Response in Selected Language")
            response_english = gr.Textbox(label="Response in English")

        # Create a row with text input and response output
        with gr.Row():
            question = gr.Textbox(label="Enter your question", placeholder="Type your question here")
            country = gr.Radio(["France", "Germany", "Italy", "Spain"], label="Language", info="Where language do you want?")

        # Create buttons to trigger transcription and response generation
        transcribe_button = gr.Button("Transcribe and Get Response")
        text_button = gr.Button("Get Response")

        # Define the button click events
        transcribe_button.click(fn=handle_voice_input, inputs=[audio, country], outputs=[transcription, response_lang, response_english])
        text_button.click(fn=handle_text_input, inputs=[question, country], outputs=[response_lang, response_english])

    # Launch the Gradio app
    demo.launch(share=False, debug=True)

if __name__ == '__main__':
    main()
