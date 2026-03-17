import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt


def visualize_spectrogram(audio_data):
    """
    Generate and visualize spectrogram from audio data

    Args:
        audio_data: tuple (sample_rate, audio_array)

    Returns:
        spectrogram image
    """
    if audio_data is None:
        return None

    sample_rate, y = audio_data

    # Ensure audio is mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Normalize audio
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    # 创建图像
    fig, ax = plt.subplots(figsize=(12, 6))

    # 显示 spectrogram
    img = librosa.display.specshow(
        D,
        sr=sample_rate,
        x_axis="time",
        y_axis="log",
        ax=ax,
    )

    ax.set_title("Spectrogram", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)

    # 添加颜色条
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("Power (dB)", fontsize=12)

    plt.tight_layout()

    return fig


def demo():
    """Create Spectrogram visualization application"""
    with gr.Blocks(title="Spectrogram Visualizer") as demo:
        gr.Markdown(
            """
            # 🎵 Spectrogram Visualizer
            
            Record audio from your microphone and visualize its spectrogram in real-time.
            This helps you understand the frequency characteristics of different audio sources, such as piano sounds.
            
            **How to use:**
            1. Click the microphone icon to start recording
            2. After recording, the spectrogram will be displayed automatically
            3. Record again to compare spectrograms of different audio sources
            """
        )

        with gr.Row():
            with gr.Column():
                # Record audio from microphone
                audio_input = gr.Audio(
                    label="Microphone Input",
                    sources=["microphone"],
                    type="numpy",
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#01C6FF",
                        skip_length=100,
                    ),
                )

                # Visualize button
                process_button = gr.Button(
                    "📊 Visualize Spectrogram", variant="primary", size="lg"
                )

            with gr.Column():
                # Display spectrogram
                spectrogram_plot = gr.Plot(label="Spectrogram", show_label=True)

        # Bind processing function
        process_button.click(
            fn=visualize_spectrogram, inputs=audio_input, outputs=spectrogram_plot
        )

        # Auto display spectrogram when audio changes
        audio_input.change(
            fn=visualize_spectrogram, inputs=audio_input, outputs=spectrogram_plot
        )

        # Add information
        gr.Markdown(
            """
            ## 📖 About Spectrogram

            A **Spectrogram** is a time-frequency representation that shows the energy distribution of an audio signal across different times and frequencies.

            - **X-axis**: Time (seconds)
            - **Y-axis**: Frequency (Hz) on a logarithmic scale
            - **Color**: Power (dB) - brighter colors indicate stronger energy

            ### 🎹 Piano Audio Features
            - **Multiple horizontal lines**: Represent the fundamental frequency and its harmonics
            - **High energy in mid-low frequencies**: Most piano energy is concentrated in this range
            - **Decay over time**: Piano notes gradually fade out
            """
        )

    return demo


if __name__ == "__main__":
    app = demo()
    app.launch(share=False, theme=gr.themes.Soft())
