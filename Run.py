import asyncio
import concurrent.futures
import itertools
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
import traceback # Added for detailed error logging

import customtkinter as ctk
from PIL import Image
# from tqdm import tqdm # tqdm in GUI needs special handling or removal
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

# To potentially silence TensorFlow/oneDNN informational messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Configuration ---
MODEL_NAME = "Salesforce/blip-image-captioning-large"
DEFAULT_MAX_CAPTION_LENGTH = 50
DEFAULT_INFERENCE_BATCH_SIZE = 16
DEFAULT_PROCESSING_CHUNK_SIZE = 50000

# --- Helper Functions ---
def get_image_files(folder_path):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    return image_files

def get_caption_filepath(image_path: Path) -> Path:
    return image_path.with_suffix('.txt')

# --- Core Captioning Logic ---
class ImageCaptioner:
    def __init__(self, device="cuda", inference_batch_size=DEFAULT_INFERENCE_BATCH_SIZE, max_caption_length=DEFAULT_MAX_CAPTION_LENGTH):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}") # Will be logged in GUI

        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            # print(f"Attempting to load model with dtype: {dtype}") # Logged in GUI
            self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(self.device)
        except Exception as e:
            # print(f"Failed to load model with specified dtype: {e}. Falling back to default.") # Logged in GUI
            self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)
        
        self.inference_batch_size = inference_batch_size
        self.max_caption_length = max_caption_length
        self.model.eval()

    async def caption_batch(self, image_paths_batch):
        if not image_paths_batch:
            return []
        images_pil = []
        valid_paths = []
        for img_path in image_paths_batch:
            try:
                img = Image.open(img_path).convert("RGB")
                images_pil.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}") # Log this to GUI instead
        
        if not images_pil:
            return []

        try:
            inputs = self.processor(images=images_pil, return_tensors="pt", padding=True, truncation=True).to(self.device, getattr(self.model, 'dtype', torch.float32))
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda', dtype=getattr(self.model, 'dtype', torch.float32)):
                generated_ids = self.model.generate(**inputs, max_length=self.max_caption_length, num_beams=4, early_stopping=True)
            captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return list(zip(valid_paths, captions))
        except Exception as e:
            # print(f"Error during model inference: {e}") # Log this to GUI
            # This should be re-raised or handled to inform the calling async function
            raise  # Re-raise to be caught by process_images_async's error handling

    async def save_caption(self, image_path, caption_text, loop):
        caption_file = get_caption_filepath(image_path)
        try:
            await loop.run_in_executor(None, lambda: caption_file.write_text(caption_text.strip(), encoding='utf-8'))
        except Exception as e:
            print(f"Error saving caption for {image_path}: {e}") # Log to GUI

# --- GUI Application ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Async Image Captioner")
        self.geometry("700x550")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- Asyncio Setup ---
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        self.running_async_pump = True

        # --- Variables ---
        self.image_folder = tk.StringVar()
        self.processing_chunk_size_var = tk.IntVar(value=DEFAULT_PROCESSING_CHUNK_SIZE)
        self.inference_batch_size_var = tk.IntVar(value=DEFAULT_INFERENCE_BATCH_SIZE)
        self.max_caption_length_var = tk.IntVar(value=DEFAULT_MAX_CAPTION_LENGTH)
        self.is_processing = False
        self.captioner: ImageCaptioner = None
        self.total_images = 0
        self.processed_images = 0

        # --- UI Elements ---
        folder_frame = ctk.CTkFrame(self)
        folder_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        folder_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(folder_frame, text="Image Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.folder_entry = ctk.CTkEntry(folder_frame, textvariable=self.image_folder, width=300)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ctk.CTkButton(folder_frame, text="Browse", command=self.browse_folder)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        config_frame = ctk.CTkFrame(self)
        config_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(config_frame, text="Processing Chunk Size (UI):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.chunk_size_entry = ctk.CTkEntry(config_frame, textvariable=self.processing_chunk_size_var)
        self.chunk_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(config_frame, text="Inference Batch Size (GPU):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.inference_batch_entry = ctk.CTkEntry(config_frame, textvariable=self.inference_batch_size_var)
        self.inference_batch_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(config_frame, text="Max Caption Length:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.max_length_entry = ctk.CTkEntry(config_frame, textvariable=self.max_caption_length_var)
        self.max_length_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)
        self.start_button = ctk.CTkButton(action_frame, text="Start Captioning", command=self.start_processing_gui_event)
        self.start_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        progress_frame = ctk.CTkFrame(self)
        progress_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_bar = ctk.CTkProgressBar(progress_frame, orientation="horizontal", mode="determinate")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.status_label = ctk.CTkLabel(progress_frame, text="Status: Idle")
        self.status_label.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.log_text = ctk.CTkTextbox(progress_frame, height=150, state=tk.DISABLED)
        self.log_text.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        self.after(50, self.periodic_async_pump) # Start pumping asyncio events
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def log_message(self, message):
        if self.log_text.winfo_exists(): # Check if widget exists
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)
            self.update_idletasks()

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.image_folder.set(folder_selected)
            self.log_message(f"Selected folder: {folder_selected}")

    def update_progress(self):
        if not self.progress_bar.winfo_exists(): return

        if self.total_images > 0:
            progress_value = self.processed_images / self.total_images
            self.progress_bar.set(progress_value)
            self.status_label.configure(text=f"Status: Processing... {self.processed_images}/{self.total_images} images. ({progress_value*100:.2f}%)")
        else:
            self.progress_bar.set(0)
            current_status = self.status_label.cget("text")
            if "Error" not in current_status and "Completed" not in current_status : # Don't overwrite error/completion status
                 self.status_label.configure(text="Status: Idle")
        self.update_idletasks()
        
    def periodic_async_pump(self):
        if not self.running_async_pump:
            return
        self.loop.call_soon(self.loop.stop)
        self.loop.run_forever()
        if self.running_async_pump:
            self.after(50, self.periodic_async_pump)

    async def process_images_async(self):
        if self.is_processing:
            return

        self.is_processing = True
        self.start_button.configure(state=tk.DISABLED, text="Processing...")
        self.log_message("Initializing processing...")
        self.processed_images = 0
        self.total_images = 0
        self.update_progress() # Initialize progress display

        try:
            folder = self.image_folder.get()
            if not folder or not os.path.isdir(folder):
                self.log_message("Error: Please select a valid image folder.")
                return 

            processing_chunk_size = self.processing_chunk_size_var.get()
            inference_batch_size = self.inference_batch_size_var.get()
            max_caption_length = self.max_caption_length_var.get()

            if inference_batch_size <= 0:
                self.log_message("Error: Inference batch size must be greater than 0.")
                return

            self.log_message(f"Processing chunk size (UI): {processing_chunk_size}")
            self.log_message(f"Inference batch size (GPU): {inference_batch_size}")
            self.log_message(f"Max caption length: {max_caption_length}")

            try:
                self.captioner = ImageCaptioner(
                    inference_batch_size=inference_batch_size,
                    max_caption_length=max_caption_length
                )
                self.log_message(f"Image captioner initialized on device: {self.captioner.device}")
                if self.captioner.device.type == 'cpu':
                    self.log_message("Warning: CUDA not available, using CPU. Processing will be very slow.")
            except Exception as e_init:
                self.log_message(f"Error initializing captioner: {e_init}")
                self.log_message(traceback.format_exc())
                return

            self.log_message("Scanning for image files...")
            all_image_paths = get_image_files(folder)
            self.total_images = len(all_image_paths)
            self.update_progress()

            if not all_image_paths:
                self.log_message("No image files found in the selected folder.")
                return

            self.log_message(f"Found {self.total_images} images to process.")

            for i in range(0, self.total_images, processing_chunk_size):
                if not self.running_async_pump: # Check if app is closing
                    self.log_message("Processing halted due to application closing.")
                    return

                chunk_image_paths = all_image_paths[i:i + processing_chunk_size]
                self.log_message(f"Processing chunk {i // processing_chunk_size + 1} of { (self.total_images + processing_chunk_size -1) // processing_chunk_size } ({len(chunk_image_paths)} images).")

                for j in range(0, len(chunk_image_paths), self.captioner.inference_batch_size):
                    if not self.running_async_pump: # Check if app is closing
                        self.log_message("Processing halted due to application closing.")
                        return

                    inference_paths_batch = chunk_image_paths[j:j + self.captioner.inference_batch_size]
                    
                    try:
                        caption_results = await self.captioner.caption_batch(inference_paths_batch)
                    except Exception as e_batch:
                        self.log_message(f"Error during batch captioning for {len(inference_paths_batch)} images: {e_batch}")
                        self.log_message(traceback.format_exc())
                        # Mark these as processed (attempted) and continue
                        self.processed_images += len(inference_paths_batch)
                        self.update_progress()
                        continue 

                    save_tasks = []
                    for img_path, caption in caption_results:
                        save_tasks.append(self.captioner.save_caption(img_path, caption, self.loop))
                    
                    await asyncio.gather(*save_tasks)

                    self.processed_images += len(inference_paths_batch)
                    self.update_progress()
                    await asyncio.sleep(0.01) # Yield control briefly

                self.log_message(f"Finished processing chunk {i // processing_chunk_size + 1}.")
                await asyncio.sleep(0.01) # Yield control

            if self.processed_images == self.total_images:
                 self.log_message("All images processed successfully!")
                 self.status_label.configure(text=f"Status: Completed {self.total_images}/{self.total_images} images.")
            else:
                 self.log_message(f"Processing finished. {self.processed_images}/{self.total_images} processed.")
                 self.status_label.configure(text=f"Status: Partial completion {self.processed_images}/{self.total_images}.")


        except Exception as e_outer:
            self.log_message(f"An unexpected error occurred in processing: {e_outer}")
            self.log_message(traceback.format_exc())
            if self.status_label.winfo_exists(): self.status_label.configure(text="Status: Error occurred.")
        finally:
            self.is_processing = False
            if self.start_button.winfo_exists(): self.start_button.configure(state=tk.NORMAL, text="Start Captioning")
            self.update_progress() # Final progress update

    def start_processing_gui_event(self):
        if self.is_processing:
            self.log_message("Processing is already in progress.")
            return
        self.loop.create_task(self.process_images_async())

    def on_closing(self):
        self.log_message("Closing application...")
        self.running_async_pump = False # Stop the pump

        tasks_to_cancel = [task for task in asyncio.all_tasks(loop=self.loop) if task is not asyncio.current_task(loop=self.loop)]
        if tasks_to_cancel:
            self.log_message(f"Cancelling {len(tasks_to_cancel)} outstanding tasks...")
            for task in tasks_to_cancel:
                task.cancel()
            
            async def wait_for_cancellation():
                try:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                    self.log_message("Background tasks cancelled or completed.")
                except asyncio.CancelledError:
                     self.log_message("Cancellation processed during shutdown.") # Should be caught by return_exceptions

            self.loop.run_until_complete(wait_for_cancellation())

        self.log_message("Closing asyncio loop...")
        if not self.loop.is_closed():
            self.loop.close()
        
        # Clean up captioner explicitly to release GPU memory
        if hasattr(self, 'captioner') and self.captioner:
            if hasattr(self.captioner, 'model'): del self.captioner.model
            if hasattr(self.captioner, 'processor'): del self.captioner.processor
            self.captioner = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log_message("Captioner resources and CUDA cache released.")
        
        self.destroy()

# --- Main ---
if __name__ == "__main__":
    # Optional: Set event loop policy for Windows if needed (usually not required for this setup anymore)
    # if os.name == 'nt' and sys.version_info >= (3,8): # ProactorEventLoop is default on Py3.8+ for asyncio on Windows
    #    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    app = App()
    app.mainloop()