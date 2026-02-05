# Plan

Duration: 1.5 hours

Workshop Flow:

1. Introduction (10 min)
2. Automatic Speech Recognition (ASR) Technology
3. Text To Speech (TTS) Technology
4. Cutting Edge Technologies



## Introduction

Start with a live runnable Demo: Voice wake-up (e.g. "Xiao Ai"/ "Jarvis"), voice command (e.g. ask it to translate), voice output (AI speaks the translated result)

Use PPT to explain the underlying workflow of the entire demo (VAD → ASR → LLM/NLP → TTS)

Briefly introduce the historical development of voice interaction technologies.



## ASR

1. Introduce basic audio concepts (sample rate, bit depth)  
   [Goal: Help non-technical students understand how audio is stored and captured]
2. Spectrogram (Explain what is the most important representation in speech processing and why it matters)
3. Deep dive into how spectrograms are created (introduce the following concepts):
   1. Fourier Transform (FFT)
   2. Mel Scale / Mel Spectrogram
   3. MFCC and Filterbanks
4. What are the mainstream models nowadays, and what are their core ideas? (CTC loss, Transformer/Conformer)
5. Show a piece of code in Colab that calls e.g. Hugging Face or OpenAI service (teach participants how to use them)



## TTS

1. Introduce the evolution path of TTS technology (if possible, compare using three audio clips or live demos):
   1. Early concatenation-based methods
   2. Mid-period parametric synthesis
   3. Modern deep learning-based approaches
2. TTS implementation pipeline:
   1. Text frontend (preprocessing)
   2. Acoustic model
   3. Vocoder
3. Show a piece of code calling a TTS service, demonstrating results with different parameters (emotion, speech rate/length, etc.)



## Future

Explain some current limitations of mainstream technology (ASR → NLP → TTS pipeline): high latency, information loss

Introduce that current mainstream research is moving toward fully End-to-End models (speech input directly to speech output)

Finally, provide some learning resources:
- Different models on Hugging Face
- Some classic papers
- How to use popular open-source frameworks (with pseudo-code or simple code examples)



## Requirement For This Workshop

1. A live demo (Runnable app) in the introduction
2. PPT
3. Piece of code calling ASR service
4. Piece of code calling TTS service