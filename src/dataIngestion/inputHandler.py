import os

class InputHandler:
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
        print("Handling audio file:", self.data)
        
        return f"Processed audio file: {self.data}"

    def videoHandler(self):
        """
        Processes a video file.
        Returns:
            str: A message indicating the video file was processed.
        """
        print("Handling video file:", self.data)
        
        return f"Processed video file: {self.data}"

    def documentHandler(self):
        """
        Processes a document file.
        Returns:
            str: A message indicating the document file was processed.
        """
        print("Handling document file:", self.data)
       
        return f"Processed document file: {self.data}"

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

if __name__ == "__main__":

    test_files = ["song.mp3", "movie.mp4", "report.pdf", "unknown.xyz"]

    for file in test_files:
        print("\nProcessing file:", file)
        handler = InputHandler(file)
        result = handler.process()
        print("Result:", result)
