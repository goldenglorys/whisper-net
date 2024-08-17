import gradio as gr
import whisper
import torch
from langchain_community.llms import Ollama

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file, sr=16000)
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    result = model.transcribe(audio_tensor, fp16=False)['text']
    return result

# Function to generate responses using the LLM
def generate_response(question):
    llm = Ollama(model="llama3:8b")
    response = llm.invoke(question)
    return response

# Function to handle voice input
def handle_voice_input(audio_file):
    transcription = transcribe_audio(audio_file)
    response = generate_response(transcription)
    return transcription, response

# Function to handle text input
def handle_text_input(question):
    response = generate_response(question)
    return response

# Function to launch the Gradio application
def main():
    # Create a Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice and Text Assistant")
        gr.Markdown("### Ask a question by voice or text and get a response.")

        # Create a row with audio input and transcription output
        with gr.Row():
            audio = gr.Audio(sources=["microphone"], label="Record your voice", type="filepath", max_length=10)
            transcription = gr.Textbox(label="Transcription")
            response = gr.Textbox(label="Response")

        # Create a row with text input and response output
        with gr.Row():
            question = gr.Textbox(label="Enter your question", placeholder="Type your question here")

        # Create buttons to trigger transcription and response generation
        transcribe_button = gr.Button("Transcribe and Get Response")
        text_button = gr.Button("Get Response")

        # Define the button click events
        transcribe_button.click(fn=handle_voice_input, inputs=audio, outputs=[transcription, response])
        text_button.click(fn=handle_text_input, inputs=question, outputs=response)

    # Launch the Gradio app
    demo.launch(share=False, debug=True)

if __name__ == '__main__':
    main()
