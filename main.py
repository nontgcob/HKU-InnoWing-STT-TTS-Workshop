import sounddevice as sd
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch
import sys
from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan

load_dotenv(find_dotenv())

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FFMPEG_INPUT_DEVICE = os.getenv("FFMPEG_INPUT_DEVICE", "default")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
SPEAKER_ID = int(os.getenv("SPEAKER_ID", "7000"))

print(f"Using device: {DEVICE}")
print(f"Using model: {MODEL_NAME}")
print(f"Using speaker ID: {SPEAKER_ID}")
print(f"Using ffmpeg input device: {FFMPEG_INPUT_DEVICE}")
# ===========================
# INITIALIZATION
# ===========================

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_API_BASE_URL"),
)

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=DEVICE
)

transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device=DEVICE
)

embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True
)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)


def check_environment_variables():
    required_vars = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_NAME"]
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Required environment variable {var} not set.")


def launch_fn(
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
        ffmpeg_input_device=FFMPEG_INPUT_DEVICE,
    )

    print("Listening for wake word...")
    for audio_chunk in mic:
        predictions = classifier(audio_chunk["raw"])
        prediction = predictions[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                return True


def transcribe(chunk_length_s=5.0, stream_chunk_s=1, debug=False) -> str:
    sampling_rate = transcriber.feature_extractor.sampling_rate

    print("Start speaking...")
    for audio_chunk in ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
        ffmpeg_input_device=FFMPEG_INPUT_DEVICE,
    ):
        audio_data = {
            "array": audio_chunk["raw"],
            "sampling_rate": sampling_rate,
        }

        item = transcriber(audio_data, chunk_length_s=30, ignore_warning=True)
        if debug:
            sys.stdout.write("\033[K")
            print(item["text"], end="\r")
        if not audio_chunk.get("partial", False):
            return item["text"]


def query(text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a voice assistant, please answer concisely and do not output markdown",
            },
            {"role": "user", "content": text},
        ],
        max_tokens=2048,
        temperature=0.7,
    )
    return response.choices[0].message.content


def synthesise(text, speaker_id: int = SPEAKER_ID):
    if speaker_id < 0 or speaker_id >= len(embeddings_dataset):
        raise ValueError(
            f"Speaker ID must be between 0 and {len(embeddings_dataset) - 1}"
        )
    speaker_embeddings = torch.tensor(
        embeddings_dataset[speaker_id]["xvector"]
    ).unsqueeze(0)
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(DEVICE), speaker_embeddings.to(DEVICE), vocoder=vocoder
    )
    return speech.cpu().numpy()


def voice_assistant():
    while True:
        launch_fn(debug=True)
        speech = synthesise("Yes Sir, how can I help you?")
        sd.play(speech, samplerate=16000)
        sd.wait()

        transcription = transcribe(debug=True)
        if any(
            word in transcription.strip().lower()
            for word in ["exit", "quit", "goodbye"]
        ):
            speech = synthesise("Goodbye! hope you have a great day!")
            sd.play(speech, samplerate=16000)
            sd.wait()
            print("Exiting voice assistant.")
            break
        print()
        print("Thinking...")
        response = query(transcription)
        print(response)
        speech = synthesise(response)
        sd.play(speech, samplerate=16000)
        sd.wait()


if __name__ == "__main__":
    voice_assistant()
