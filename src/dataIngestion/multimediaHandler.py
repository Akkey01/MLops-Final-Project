import os
import time
from whisper import getAudioFileFromVideo, audio_to_text
from PyPDF2 import PdfReader


class MultiMediaHandler:
    def __init__(self, data):
        self.data = data

    def checkFileType(self):
        """
        Determines the file type based on the file extension.
        Returns:
            str: 'audio', 'video', 'document', or 'unknown'
        """
        if isinstance(self.data, str):
            _, ext = os.path.splitext(self.data)
            ext = ext.lower()
            if ext in ['.mp3', '.wav', '.aac']:
                return 'audio'
            elif ext in ['.mp4', '.avi', '.mov']:
                return 'video'
            elif ext in ['.pdf', '.doc', '.docx', '.txt']:
                return 'document'
            
        return 'unknown'

    def audioHandler(self):
        """
        Processes an audio file.
        Returns:
            str: A message indicating the audio file was processed.
        """
        custom_text_filename = f'{time.now}.txt'
        audio_to_text(self.data, custom_text_filename)

    def videoHandler(self):
        """
        Processes a video file.
        Returns:
            str: A message indicating the video file was processed.
        """
        custom_audio_filename = f'{time.now}.wav'
        custom_text_filename = f'{time.now}.txt'
        getAudioFileFromVideo(self.data, custom_audio_filename)
        audio_to_text(custom_audio_filename, custom_text_filename)

    def documentHandler(self):
        """
        Processes a document file.
        Returns:
            str: A message indicating the document file was processed.
        """
        reader = PdfReader(self.data)

        all_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

        output_file = f'{time.now}.txt'

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)

    def process(self):
        """
        Determines the file type and processes it with the corresponding handler.
        Returns:
            str: The result of the processing or a message if the file type is unsupported.
        """
        file_type = self.checkFileType()
        if file_type == 'audio':
            return self.audioHandler()
        elif file_type == 'video':
            return self.videoHandler()
        elif file_type == 'document':
            return self.documentHandler()
        else:
            print("Unsupported file type:", self.data)
            return "Unsupported file type"