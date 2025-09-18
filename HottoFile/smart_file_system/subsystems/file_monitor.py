import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class FileMonitor:
    def __init__(self, path, handler_func, watch_dirs=False):
        self.path = path
        self.handler_func = handler_func
        self.watch_dirs = watch_dirs
        self.event_handler = self.NewFileHandler(handler_func, watch_dirs)
        self.observer = Observer()

    def start(self):
        self.observer.schedule(self.event_handler, self.path, recursive=True)
        self.observer.start()
        logging.info(f"Monitoring started on: {self.path}")

    def stop(self):
        self.observer.stop()
        self.observer.join()
        logging.info("Monitoring stopped.")

    class NewFileHandler(FileSystemEventHandler):
        def __init__(self, handler_func, watch_dirs):
            self.handler_func = handler_func
            self.watch_dirs = watch_dirs

        def on_created(self, event):
            if not self.watch_dirs and event.is_directory:
                return
            # 等待写入完成
            time.sleep(0.5)
            if os.path.exists(event.src_path):
                logging.info(f"New file detected: {event.src_path}")
                self.handler_func(event.src_path)
