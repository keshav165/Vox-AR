import numpy as np
import sounddevice as sd
import tempfile
from text_to_3d_generator import generate_3d_model
import os
import wave
import json

def record_audio(duration=5):
    """
    Record audio from microphone for a specified duration.
    
    Args:
        duration (float): Recording duration in seconds
        
    Returns:
        numpy.ndarray: Audio data
    """
    print(f"Recording for {duration} seconds...")
    sample_rate = 16000  # Vosk's preferred sample rate
    audio_data = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1)
    sd.wait()  # Wait until recording is complete
    return audio_data

def transcribe_audio(audio_data):
    """
    Transcribe audio data using Vosk.
    
    Args:
        audio_data (numpy.ndarray): Audio data to transcribe
        
    Returns:
        str: Transcribed text
    """
    # Create a temporary WAV file
    temp_filename = None
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_filename = temp_file.name
        temp_file.close()  # Close the file handle
        
        # Save audio data as WAV file
        with wave.open(temp_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data.tobytes())
            
        # Initialize Vosk model and recognizer
        model = Model("vosk-model-small-en-us-0.15")
        recognizer = KaldiRecognizer(model, 16000)
        
        # Read audio data
        with open(temp_filename, "rb") as audio_file:
            audio_bytes = audio_file.read()
            
        # Process audio in chunks
        if recognizer.AcceptWaveform(audio_bytes):
            result = recognizer.Result()
            text = json.loads(result).get('text', '')
        else:
            partial = recognizer.PartialResult()
            text = json.loads(partial).get('partial', '')
            
        return text.strip()
        
    finally:
        # Ensure file is deleted even if an error occurs
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                print(f"Warning: Could not delete temporary file {temp_filename}")

def voice_to_3d():
    """
    Main function to handle voice input and generate 3D model.
    """
    # Create output directory if it doesn't exist
    output_dir = "generated_models"
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        # Record and transcribe voice
        print("\nPlease speak your prompt when ready...")
        audio_data = record_audio()
        prompt = transcribe_audio(audio_data)
        
        if prompt:
            print(f"Recognized text: {prompt}")
            # Generate 3D model using the recognized text
            gif_path, ply_path, obj_path = generate_3d_model(prompt, output_dir)
            print(f"\nGeneration complete!")
            print(f"GIF saved to: {gif_path}")
            print(f"PLY saved to: {ply_path}")
            print(f"OBJ saved to: {obj_path}")
            
            # Ask if user wants to generate another model
            response = input("\nWould you like to generate another model? (y/n): ")
            if response.lower() != 'y':
                break
        else:
            print("No valid input received. Please try again.")

if __name__ == "__main__":
    voice_to_3d()
