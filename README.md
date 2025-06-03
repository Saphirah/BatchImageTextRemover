# Image Inpainting with Simple-Lama and OCR

This script downloads images, detects text regions using Tesseract OCR, and removes the text using the Simple-Lama inpainting model.

## Features

* Downloads images from URLs in `images.json`
* Detects text via `pytesseract`
* Creates binary masks of text regions
* Inpaints text using Simple-Lama
* Saves results to `target/` folder

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Ensure `tesseract` is available in your system path.

## Usage

1. Place `images.json` in the root directory with entries like:

   ```json
   [{"id": 1, "url": "https://example.com/image.jpg"}]
   ```

2. Run the script:

   ```bash
   python main.py
   ```

## Output

* Downloads images to `source/`
* Saves inpainted images to `target/`
