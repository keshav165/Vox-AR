import numpy as np
import sounddevice as sd
import tempfile
import os
import wave
import speech_recognition as sr
from text_to_3d_generator import generate_3d_model
import config

def record_audio(duration=None):
    duration = duration or config.DURATION
    sample_rate = config.SAMPLE_RATE
    print(f"Recording for {duration} seconds...")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None
    temp_filename = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        with wave.open(temp_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None
    return temp_filename

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text.strip()
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Speech Recognition request failed: {e}")
    except Exception as e:
        print(f"Error during transcription: {e}")
    return ""

def voice_to_3d():
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    while True:
        print("\nPlease speak your prompt when ready...")
        audio_path = record_audio()
        if not audio_path:
            print("Audio recording failed. Please check your microphone and try again.")
            continue
        prompt = transcribe_audio(audio_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if prompt:
            print(f"Recognized text: {prompt}")
            try:
                gif_path, ply_path, obj_path = generate_3d_model(prompt, output_dir)
                print("\nGeneration complete!")
                print(f"GIF saved to: {gif_path}")
                print(f"PLY saved to: {ply_path}")
                print(f"OBJ saved to: {obj_path}")
            except Exception as e:
                print(f"Error generating 3D model: {e}")
                continue
            response = input("\nWould you like to generate another model? (y/n): ")
            if response.lower() != 'y':
                break
        else:
            print("No valid input received. Please try again.")

if __name__ == "__main__":
    voice_to_3d()
