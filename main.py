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


def transcribe_audio(audio_file):
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        str: Transcribed text.
    """
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file, sr=16000)
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    result = model.transcribe(audio_tensor, fp16=False)["text"]
    return result


def generate_response(question):
    """
    Generate a response using the LLM.

    Args:
        question (str): The input question.

    Returns:
        str: The generated response.
    """
    llm = Ollama(model="llama3:8b")
    response = llm.invoke(question)
    return response


def text_to_speech(text):
    """
    Convert text to speech using Parler-TTS.

    Args:
        text (str): The input text.

    Returns:
        tuple: A tuple containing the audio array and the sampling rate.
    """
    description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, model.config.sampling_rate


def handle_voice_input(audio_file):
    """
    Handle voice input, transcribe it, generate a response, and convert the response to speech.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        tuple: A tuple containing the transcription, response, and audio response.
    """
    transcription = transcribe_audio(audio_file)
    response = generate_response(transcription)
    audio_response, sampling_rate = text_to_speech(response)
    return transcription, response, (sampling_rate, audio_response)


def handle_text_input(question):
    """
    Handle text input, generate a response, and convert the response to speech.

    Args:
        question (str): The input question.

    Returns:
        tuple: A tuple containing the response and audio response.
    """
    response = generate_response(question)
    audio_response, sampling_rate = text_to_speech(response)
    return response, (sampling_rate, audio_response)


def main():
    """
    Launch the Gradio application.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Whisper-Net")
        gr.Markdown("### Soft spoken Ai that listens and understands - Ask a question by voice or text and get a response.")

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

        with gr.Row():
            question = gr.Textbox(
                label="Enter your question", placeholder="Type your question here"
            )

        transcribe_button = gr.Button("Transcribe and Get Response")
        text_button = gr.Button("Get Response")

        transcribe_button.click(
            fn=handle_voice_input,
            inputs=audio,
            outputs=[transcription, response, audio_response],
        )
        text_button.click(
            fn=handle_text_input, inputs=question, outputs=[response, audio_response]
        )

    demo.launch(share=False, debug=True)


if __name__ == "__main__":
    main()
