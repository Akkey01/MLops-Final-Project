import whisper
import torch 
import os
from moviepy.editor import VideoFileClip

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model = whisper.load_model("large", device=device)

def getAudioFileFromVideo(input , output):

    video_clip = VideoFileClip(input)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output)
    audio_clip.close()
    video_clip.close()
    print("Audio extraction successful!")