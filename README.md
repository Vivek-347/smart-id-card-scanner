# Smart ID Card Scanner API

A FastAPI project for scanning ID card images, extracting text with EasyOCR, heuristically identifying common fields, optionally cropping a face, and saving scan records to SQLite.

## Features

1. **OCR improvement** — image upscaling, denoising, CLAHE contrast enhancement, adaptive thresholding, and Otsu thresholding. OCR is run on several variants and the most text-rich result is retained.
2. **Card detection and alignment** — best-effort contour detection and perspective correction. If a card boundary is not confidently found, the original image is used.
3. **Face crop** — best-effort OpenCV Haar-cascade face detection. Face cropping is optional; unsupported OpenCV builds or no detected face will not fail the scan.
4. **SQLite storage** — records persist in `smart_id_scanner.db`.
5. **Retrieve a record** — `GET /api/v1/records/{record_id}`.
6. **List records** — `GET /api/v1/records?limit=20&offset=0`.

Extracted fields: full name, document number, date of birth, issue date, expiry date, course, blood group, and emergency contact number.

> OCR and field extraction are best-effort. Always verify extracted data against the physical card. Do not use heuristic OCR output as the sole basis for identity or eligibility decisions. This starter project stores sensitive ID information locally; restrict access and delete test scans when finished.

## Project structure

```text
smart-id-scanner-pro/
├── main.py
├── database.py
├── requirements.txt
├── README.md
├── .gitignore
├── uploads/       # original uploaded images
├── processed/     # chosen preprocessed OCR image
├── faces/         # optional face crops
└── smart_id_scanner.db  # created automatically on first run
```

## Setup on Windows PowerShell

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

If PowerShell blocks activation, use this command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open:

- Swagger UI: http://127.0.0.1:8000/docs
- Root health check: http://127.0.0.1:8000/
- List records: http://127.0.0.1:8000/api/v1/records
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

## API workflow

### 1. Scan an ID card

In Swagger, open `POST /api/v1/process-id`, click **Try it out**, choose a JPG/PNG/WEBP image, and click **Execute**.

The response includes `record_id`, extracted fields, OCR text and image metadata. The uploaded image, processed image and optional face crop are saved in project folders.

### 2. List saved records

Call `GET /api/v1/records?limit=20&offset=0`.

### 3. Retrieve one record

Copy a `record_id` from the scan response and call `GET /api/v1/records/{record_id}`.

## Important notes

- The first EasyOCR request downloads/initializes its model and may take time.
- CPU mode is supported. The PyTorch warnings about no GPU or pinned memory are normally informational.
- `opencv-python` 4.x is pinned by the requirement range because some OpenCV 5 builds may not expose `cv2.CascadeClassifier`.
- If `cv2` behaves unexpectedly, use the same Python environment for install and run:
  ```powershell
  python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
  python -m pip install --no-cache-dir "opencv-python>=4.11.0.86,<5.0"
  python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'CascadeClassifier'))"
  ```
  The last command should show a 4.x version and `True`.

## Before production

This is an interview assignment/demo, not a production identity-verification service. Add authentication, authorization, retention/deletion controls, rate limits, encrypted storage, audit logging, malware/content checks, and explicit consent before handling real ID documents.


## Recent improvements

- Date extraction supports month/year values such as `07/2025` and `06/2027`, including OCR separator noise and labels on the same or adjacent lines. Dates are only assigned when a nearby label indicates their meaning.
- Face detection tries multiple Haar cascades and contrast/upscaled image variants. Face cropping remains best-effort; some printed portraits or low-quality photos may not be detected.
- Face detection runs on the aligned card image first and falls back to the original image when card boundary detection succeeded but no face was found.
- Card boundary detection/perspective correction remains best-effort; if no reliable card quadrilateral is found, the original image is processed.
