![image](https://github.com/user-attachments/assets/78483a1b-273c-4bfa-85ad-6faebf2da4b3)

# Async BLIP Image Captioner GUI

A Python-based desktop application with a Graphical User Interface (GUI) for batch captioning images using the Salesforce BLIP-large model. It leverages asynchronous processing, CUDA for GPU acceleration, and mixed precision to optimize performance for captioning large datasets.

## Overview

This application is designed to:
-   Load images from a specified folder (including subfolders).
-   Generate descriptive captions for each image using the pre-trained `Salesforce/blip-image-captioning-large` model.
-   Save each caption as a `.txt` file with the same name as the image, in the same directory as the image.
-   Provide a user-friendly GUI for configuration and progress monitoring.
-   Utilize `asyncio` for responsive UI and `concurrent.futures` for I/O-bound tasks.
-   Employ CUDA for GPU acceleration (if available) and automatic mixed precision (FP16/BF16) for faster inference and reduced memory usage.

## Features

-   **Graphical User Interface:** Built with `customtkinter` for a modern look and feel.
-   **Batch Processing:** Efficiently handles large numbers of images.
-   **Recursive Image Search:** Finds images in the selected folder and its subdirectories.
-   **BLIP Model Integration:** Uses Hugging Face Transformers to run the powerful `Salesforce/blip-image-captioning-large` model.
-   **CUDA Acceleration:** Automatically uses NVIDIA GPUs if available for significantly faster processing.
-   **Mixed Precision (FP16/BF16):** Reduces memory footprint and speeds up inference on compatible GPUs.
-   **Asynchronous Operations:** Keeps the GUI responsive during processing and handles file I/O efficiently.
-   **Configurable Parameters:**
    -   Image Folder
    -   Processing Chunk Size (for UI management)
    -   Inference Batch Size (critical for GPU VRAM and speed)
    -   Maximum Caption Length
-   **Progress Monitoring:** Real-time progress bar and status updates.
-   **Logging:** Detailed logs displayed within the GUI for events, errors, and progress.
-   **XFormers Support (Optional):** Code includes commented-out sections to enable xformers for potentially better memory efficiency and speed (if installed).

## Requirements

-   Python 3.8+
-   PyTorch (with CUDA support for GPU acceleration)
-   Transformers (by Hugging Face)
-   Pillow (PIL)
-   customtkinter
-   tqdm (optional, primarily for command-line progress but good to have)
-   sentencepiece (often a dependency for BLIP's tokenizer)
-   xformers (optional, for enhanced performance)

## Setup and Installation

1.  **Clone the Repository (Example):**
    ```bash
    git clone https://github.com/yourusername/async-blip-captioner.git
    cd async-blip-captioner
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install PyTorch with CUDA:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system and CUDA version. For example, for CUDA 11.8:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    If you don't have an NVIDIA GPU or don't want to use CUDA, you can install the CPU-only version (processing will be very slow):
    ```bash
    pip install torch torchvision torchaudio
    ```

4.  **Install Other Dependencies:**
    You can create a `requirements.txt` file with the following content:
    ```txt
    transformers
    Pillow
    customtkinter
    # tqdm # If you decide to use it beyond the GUI
    sentencepiece
    # xformers # If you want to try xformers
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    Or install individually:
    ```bash
    pip install transformers Pillow customtkinter sentencepiece
    # Optionally, for xformers:
    # pip install xformers
    ```

## How to Run

Execute the main Python script:
```bash
Run.py 
