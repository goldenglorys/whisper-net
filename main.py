import gradio as gr
import whisper
import torch
from langchain_community.llms import Ollama
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Initialize Parler-TTS model and tokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ylacombe/parler-tts-mini-jenny-30H"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("ylacombe/parler-tts-mini-jenny-30H")


# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file, sr=16000)
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    result = model.transcribe(audio_tensor, fp16=False)["text"]
    return result


# Function to generate responses using the LLM
def generate_response(question):
    llm = Ollama(model="llama3:8b")
    response = llm.invoke(question)
    return response


# Function to convert text to speech using Parler-TTS
def text_to_speech(text):
    description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, model.config.sampling_rate


# Function to handle voice input
def handle_voice_input(audio_file):
    transcription = transcribe_audio(audio_file)
    response = generate_response(transcription)
    audio_response, sampling_rate = text_to_speech(response)
    return transcription, response, (sampling_rate, audio_response)


# Function to handle text input
def handle_text_input(question):
    response = generate_response(question)
    audio_response, sampling_rate = text_to_speech(response)
    return response, (sampling_rate, audio_response)


# Function to launch the Gradio application
def main():
    # Create a Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice and Text Assistant")
        gr.Markdown("### Ask a question by voice or text and get a response.")

        # Create a row with audio input and transcription output
        with gr.Row():
            audio = gr.Audio(
                sources=["microphone"],
                label="Record your voice",
                type="filepath",
                max_length=10,
            )
            transcription = gr.Textbox(label="Transcription")
            response = gr.Textbox(label="Response")
            audio_response = gr.Audio(label="Audio Response")

        # Create a row with text input and response output
        with gr.Row():
            question = gr.Textbox(
                label="Enter your question", placeholder="Type your question here"
            )

        # Create buttons to trigger transcription and response generation
        transcribe_button = gr.Button("Transcribe and Get Response")
        text_button = gr.Button("Get Response")

        # Define the button click events
        transcribe_button.click(
            fn=handle_voice_input,
            inputs=audio,
            outputs=[transcription, response, audio_response],
        )
        text_button.click(
            fn=handle_text_input, inputs=question, outputs=[response, audio_response]
        )

    # Launch the Gradio app
    demo.launch(share=False, debug=True)


if __name__ == "__main__":
    main()
