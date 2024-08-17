import gradio as gr
import whisper
import torch

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file, sr=16000)
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    result = model.transcribe(audio_tensor, fp16=False)['text']
    return result

# Function to launch the Gradio application
def main():
    # Create a Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice Transcription App")
        gr.Markdown("### Record your voice and get the transcription.")

        # Create a row with audio input and transcription output
        with gr.Row():
            audio = gr.Audio(sources=["microphone"], label="Record your voice", type="filepath", max_length=10)
            transcription = gr.Textbox(label="Transcription")

        # Create a button to trigger transcription
        transcribe_button = gr.Button("Transcribe")

        # Define the button click event
        transcribe_button.click(fn=transcribe_audio, inputs=audio, outputs=transcription)

    # Launch the Gradio app
    demo.launch(share=False, debug=True)

if __name__ == '__main__':
    main()
