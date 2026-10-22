SMART ID CARD SCANNER - QUICK PATCH

This patch updates main.py to:
1. Run a second OCR-tolerant date extraction pass for labels such as "Issied on" and "Valid Uplo".
2. Upscale small ID-card portraits before OpenCV Haar face detection.
3. Retry face detection on the original image if the card-cropped image did not yield a face.
4. Return metadata.face_detected for quick debugging.

IMPORTANT: Keep your existing database.py in the same folder as main.py.
This patch does not replace your database or existing records.

INSTALL / REPAIR OPENCV (PowerShell, from your project folder):
  python -m pip install --force-reinstall opencv-python==4.11.0.86
  python -c "import cv2; print(cv2.__version__); print('CascadeClassifier:', hasattr(cv2, 'CascadeClassifier')); print('haar:', hasattr(cv2, 'data'))"

Expected: version 4.11.0.86, CascadeClassifier: True, haar: True

RUN:
  python -m uvicorn main:app --reload

Then test POST /api/v1/process-id at http://127.0.0.1:8000/docs.
If the API is already running, stop it with Ctrl+C before restarting.

Note: OCR and face detection are best-effort. Always verify extracted dates against the card.
