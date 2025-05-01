from multimediaHandler import MultiMediaHandler
import sys

if __name__ == "__main__":

    print("Invoked")

    if len(sys.argv) < 2:
        print("Usage: python main.py <file1> <file2> ...")
        sys.exit(1)

    test_files = sys.argv[1:]  # Get all file arguments

    for file in test_files:
        print("\nProcessing file:", file)
        handler = MultiMediaHandler(file)

        result = handler.process()
<<<<<<< HEAD
        print("Result:", result)
        
=======
        print("Result:", result)
>>>>>>> 62be11b1f45ddcbfee65258a35909a2b4b05735a
