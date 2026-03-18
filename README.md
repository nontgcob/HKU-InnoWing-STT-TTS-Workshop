# HKU-InnoWing-STT-TTS-Workshop

Welcome to the Speech-to-Text (STT) & Text-to-Speech (TTS) Workshop by InnoWing, HKU.

This repository is developed by CHAMADOL Nutnornont & Ivan Xieyi Fan

Student Research Assistant Interns at Tam Wing Fan Innovation Wing, The University of Hong Kong

# Workshop Overview
Duration: 1.5 hours

Workshop Flow:

1. Introduction: Motivation & Live Demo
2. Fundamental Concepts & Coding Skills
3. Speech Generation (Text-to-Speech, TTS) Technology
4. Speech Recognition (Speech-to-Text, STT) Technology
5. Handling Unseen Words in Speech Recognition

## Introduction

Start with a live runnable Demo: Voice wake-up (e.g. "Xiao Ai"/ "Jarvis"), voice command (e.g. ask it to translate), voice output (AI speaks the translated result)

Explain the underlying workflow of the entire demo (VAD → ASR → LLM/NLP → TTS)

Briefly introduce the historical development of voice interaction technologies.

## Speech Recognition (STT)

1. Introduce basic audio concepts (sample rate, bit depth)  
   [Goal: Help non-technical students understand how audio is stored and captured]
2. Spectrogram (Explain what is the most important representation in speech processing and why it matters)
3. Deep dive into how spectrograms are created (introduce the following concepts):
   1. Fourier Transform (FFT)
   2. Mel Scale / Mel Spectrogram
   3. MFCC and Filterbanks
4. What are the mainstream models nowadays, and what are their core ideas? (CTC loss, Transformer/Conformer)
5. Show a piece of code in Colab that calls e.g. Hugging Face or OpenAI service (teach participants how to use them)

## Speech Generation (TTS)

1. Introduce the evolution path of TTS technology (if possible, compare using three audio clips or live demos):
   1. Early concatenation-based methods
   2. Mid-period parametric synthesis
   3. Modern deep learning-based approaches
2. TTS implementation pipeline:
   1. Text frontend (preprocessing)
   2. Acoustic model
   3. Vocoder
3. Show a piece of code calling a TTS service, demonstrating results with different parameters (emotion, speech rate/length, etc.)

## Handling Unseen Words in Speech Recognition

Explain some current limitations of mainstream technology (STT → NLP → TTS pipeline): high latency, information loss

Introduce that current mainstream research is moving toward fully End-to-End models (speech input directly to speech output)

Finally, provide some learning resources:
- Different models on Hugging Face
- Some classic papers
- How to use popular open-source frameworks (with pseudo-code or simple code examples)

---
For the demo, please run the following

```bash
uv sync
```

```bash
uv run main.py
```

## Set Environment Variables
```
OPENAI_API_KEY=<your_key>
OPENAI_BASE_URL=<base_url>
MODEL_NAME=<name>
FFMPEG_INPUT_DEVICE="default" (specify audio device to use on your device)
SPEAKER_ID=7000 (specify voice id you want to use)
```

Say `marvin` to wake the voice assistant, and try talk to her!!!

Say `Goodbay` or `quit` to exit voice assistant.
