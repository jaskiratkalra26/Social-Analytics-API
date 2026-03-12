# OCR Setup Instructions

The OCR features in this project rely on Tesseract-OCR, an external optical character recognition engine.

## Installation Steps (Windows)

1.  **Download the Installer**:
    Go to the [UB Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki) and download the latest installer (e.g., `tesseract-ocr-w64-setup-v5.x.x.exe`).

2.  **Run the Installer**:
    Install Tesseract. Note the installation path (default is usually `C:\Program Files\Tesseract-OCR`).
    
    *Important*: During installation, you can install additional language data if needed. English is included by default.

3.  **Add to PATH (Recommended)**:
    Add the installation directory (e.g., `C:\Program Files\Tesseract-OCR`) to your system's `PATH` environment variable.
    
    *   Search for "Edit the system environment variables" in Start Menu.
    *   Click "Environment Variables".
    *   Under "System variables", find `Path` and click "Edit".
    *   Click "New" and paste the path to the folder containing `tesseract.exe`.
    *   Click OK to save.
    *   Restart your terminal/VS Code for changes to take effect.

4.  **Alternative Configuration**:
    If you cannot modify the system PATH, you can configure the path in `Config.py`:
    
    Open `Config.py` and set `TESSERACT_CMD`:
    
    ```python
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```

## Verification

After installation, you can verify it works by running:

```bash
tesseract --version
```
in a new terminal window.

Then run the provided test script:

```bash
python test_ocr_features.py
```
