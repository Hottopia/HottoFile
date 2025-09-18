
import os
import time
from subsystems.file_parser import FileParser
from subsystems.embedding_manager import EmbeddingManager
from subsystems.classifier import LLMClassifier, Classifier
from subsystems.feedback_manager import FeedbackManager
from subsystems.file_monitor import FileMonitor

# --- CONFIGURATION ---
# IMPORTANT: Set your OpenAI API key as an environment variable named OPENAI_API_KEY
# You can also set OPENAI_API_BASE if you are using a custom endpoint.
MONITORED_DIRECTORY = os.path.join(os.path.dirname(__file__), "monitored_files")

# --- INITIALIZATION ---
try:
    llm_classifier = LLMClassifier()
    file_parser = FileParser(llm=llm_classifier.client)
except ValueError as e:
    print(f"Error initializing LLMClassifier: {e}")
    print("Please make sure your OPENAI_API_KEY environment variable is set.")
    exit(1)

embedding_manager = EmbeddingManager()
classifier = Classifier(llm_classifier)
feedback_manager = FeedbackManager()

processed_files = {}

# --- CORE WORKFLOW ---
def process_new_file(file_path):
    print(f"--- Processing new file: {file_path} ---")
    
    # 1. Parse the file
    file_info = file_parser.parse_file(file_path)
    if not file_info:
        return

    # 2. Classify the file
    file_info = classifier.classify(file_info, embedding_manager)

    # 3. Add to embedding manager
    if file_info.content.get('text_summary'):
        embedding_manager.add_file(file_info)

    # Store for later access
    processed_files[file_info.file_id] = file_info

    print(f"--- Finished processing: {file_info.name} ---")
    print(f"  - File ID: {file_info.file_id}")
    print(f"  - Type: {file_info.type}")
    print(f"  - Summary: {file_info.content.get('text_summary', 'N/A')}")
    print(f"  - Final Label: {file_info.final_label}")
    print("-----------------------------------------")

# --- MAIN APPLICATION ---
def main():
    print("--- Smart File System Initializing ---")

    # Create monitored directory if it doesn't exist
    if not os.path.exists(MONITORED_DIRECTORY):
        os.makedirs(MONITORED_DIRECTORY)
        print(f"Created monitored directory: {MONITORED_DIRECTORY}")

    # Initialize and start the file monitor
    file_monitor = FileMonitor(MONITORED_DIRECTORY, process_new_file)
    file_monitor.start()

    print("--- System Ready ---Dropped files into the 'monitored_files' folder to begin.")

    try:
        while True:
            print("\nAvailable commands: [search, feedback, exit]")
            command = input("> ").strip().lower()

            if command == 'search':
                query = input("Enter search query: ").strip()
                if query:
                    results = embedding_manager.search(query, k=3)
                    if results:
                        print("\n--- Search Results ---")
                        for res in results:
                            print(f"  - File: {res['file'].name} (ID: {res['file'].file_id}), Distance: {res['distance']:.4f}")
                    else:
                        print("No similar files found.")
            
            elif command == 'feedback':
                file_id = input("Enter file ID to provide feedback for: ").strip()
                if file_id in processed_files:
                    correct_label = input(f"Enter the correct label for {processed_files[file_id].name}: ").strip()
                    if correct_label:
                        feedback_manager.add_feedback(processed_files[file_id], correct_label)
                    else:
                        print("Label cannot be empty.")
                else:
                    print("File ID not found.")

            elif command == 'exit':
                break
            
            else:
                print("Unknown command.")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        file_monitor.stop()
        print("System shut down.")

if __name__ == "__main__":
    main()
