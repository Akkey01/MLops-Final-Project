import whisper
import torch 
import os
from moviepy import VideoFileClip
import speech_recognition as sr

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model = whisper.load_model("large", device=device)

# Save output file only with .wav for the below function
def getAudioFileFromVideo(input, output):

    video_clip = VideoFileClip(input)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output)
    audio_clip.close()
    video_clip.close()
    print("Audio extraction successful!")

def audio_to_text(audio_file_path, output_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        with open(output_file_path, "w") as file:
            file.write(text)
        return file
    except sr.UnknownValueError:
        print("Could not understand audio")
        return False
    except sr.RequestError as e:
        print(f"Could not request results from speech recognition service; {e}")
        return False