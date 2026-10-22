"""
Smart ID Card Scanner API
Features:
1. Improved OCR with multiple preprocessing attempts
2. ID-card/document contour detection and perspective correction (best effort)
3. Face detection and optional portrait crop (best effort)
4. SQLite persistence for scanned records
5. Retrieve a record by ID
6. List records with pagination

Run:
    python -m pip install -r requirements.txt
    python -m uvicorn main:app --reload
Open:
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import re
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from database import init_db, save_record, get_record, list_records

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
FACES_DIR = BASE_DIR / "faces"

for folder in (UPLOAD_DIR, PROCESSED_DIR, FACES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Smart ID Card Scanner API",
    description="OCR-powered ID card scanning, field extraction, face cropping, and record storage.",
    version="2.0.0",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def root():
    return {
        "message": "Smart ID Card Scanner API is running",
        "version": app.version,
        "docs": "/docs",
        "endpoints": [
            "POST /api/v1/process-id",
            "GET /api/v1/records",
            "GET /api/v1/records/{record_id}",
        ],
    }


@app.get("/api/v1/records")
def get_records(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    """List saved records newest-first. Limit is capped at 100 per request."""
    records, total = list_records(limit=limit, offset=offset)
    return {"total": total, "limit": limit, "offset": offset, "records": records}


@app.get("/api/v1/records/{record_id}")
def retrieve_record(record_id: str):
    record = get_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    return record


@app.post("/api/v1/process-id")
async def process_id(file: UploadFile = File(...)):
    """Upload an ID image, preprocess it, run OCR, extract fields, and save the result."""
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WEBP image.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 10 MB.")

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

    height, width = image.shape[:2]
    record_id = str(uuid.uuid4())
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    original_path = UPLOAD_DIR / f"{record_id}{suffix}"
    processed_path = PROCESSED_DIR / f"{record_id}.png"
    original_path.write_bytes(content)

    # Step 1: Try to isolate and straighten the card. If not found, keep original.
    card_image, card_detected = detect_and_warp_card(image)

    # Step 2: Create several preprocessing variants and use the OCR result with most text.
    try:
        import easyocr
        # Cache the OCR reader instead of loading its model on every request.
        reader = get_ocr_reader(easyocr)
        ocr_text, chosen_preprocessed = run_ocr(reader, card_image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "OCR failed. Check EasyOCR installation/model availability. "
                f"Error type: {type(exc).__name__}. Details: {str(exc)[:250]}"
            ),
        )

    cv2.imwrite(str(processed_path), chosen_preprocessed)
    fields = extract_fields(ocr_text)

    # Face crop is optional. It should never fail the entire ID scan.
    face_path = detect_and_save_face(card_image, record_id)
    if face_path is None and card_image is not image:
        face_path = detect_and_save_face(image, record_id)

    processed_height, processed_width = card_image.shape[:2]
    metadata = {
        "filename": Path(file.filename or "upload").name,
        "image_width": int(width),
        "image_height": int(height),
        "processed_width": int(processed_width),
        "processed_height": int(processed_height),
        "content_type": content_type,
        "card_boundary_detected": bool(card_detected),
        "face_detected": bool(face_path),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "ocr_text": ocr_text,
        "processed_image": str(processed_path.relative_to(BASE_DIR)).replace("\\", "/"),
    }
    record = {
        "record_id": record_id,
        **fields,
        "metadata": metadata,
        "face_image": str(face_path.relative_to(BASE_DIR)).replace("\\", "/") if face_path else None,
    }
    save_record(record)

    return JSONResponse(content={"success": True, **record})


# EasyOCR's model can take a while to load, so cache one reader instance.
_OCR_READER = None

def get_ocr_reader(easyocr_module):
    global _OCR_READER
    if _OCR_READER is None:
        _OCR_READER = easyocr_module.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def detect_and_warp_card(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Detect and straighten an ID card using several contour strategies.

    The photo may include fingers, glare, and a busy background, so this function
    tries Canny edges and adaptive/OTSU thresholds. It only warps a plausible
    quadrilateral; otherwise it safely returns the original image.
    """
    try:
        h, w = image.shape[:2]
        if h < 2 or w < 2:
            return image.copy(), False

        scale = min(1.0, 1200.0 / max(h, w))
        small = cv2.resize(
            image, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        ) if scale < 1 else image.copy()
        sh, sw = small.shape[:2]
        image_area = float(sh * sw)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Different ID card backgrounds respond better to different thresholds.
        edges = cv2.Canny(blur, 35, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates = []
        for mask in (edges, adaptive, otsu, cv2.bitwise_not(adaptive), cv2.bitwise_not(otsu)):
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            candidates.extend(contours)
        candidates.sort(key=cv2.contourArea, reverse=True)

        seen = set()
        for contour in candidates[:100]:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.10 or area > image_area * 0.98:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            pts_small = approx.reshape(4, 2).astype(np.float32)
            # Skip duplicate quadrilaterals found by multiple threshold passes.
            signature = tuple(sorted((int(x // 8), int(y // 8)) for x, y in pts_small))
            if signature in seen:
                continue
            seen.add(signature)

            pts = pts_small / scale if scale < 1 else pts_small
            rect = order_points(pts)
            tl, tr, br, bl = rect
            width_a = float(np.linalg.norm(br - bl))
            width_b = float(np.linalg.norm(tr - tl))
            height_a = float(np.linalg.norm(tr - br))
            height_b = float(np.linalg.norm(tl - bl))
            out_w = int(max(width_a, width_b))
            out_h = int(max(height_a, height_b))
            if out_w < 160 or out_h < 100:
                continue
            ratio = max(out_w / max(out_h, 1), out_h / max(out_w, 1))
            # Most ID cards are landscape or portrait rectangles, not long strips.
            if ratio > 2.5:
                continue
            if min(width_a, width_b) < 0.35 * max(width_a, width_b):
                continue

            destination = np.array(
                [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), destination)
            warped = cv2.warpPerspective(
                image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if warped.size and warped.shape[0] > 0 and warped.shape[1] > 0:
                return warped, True
    except Exception:
        # Alignment is best-effort; OCR should still run on the original photo.
        pass
    return image.copy(), False


def make_preprocessing_variants(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    # Upscale small text to improve OCR, with a sensible size cap.
    h, w = gray.shape[:2]
    if max(h, w) < 1400:
        factor = min(2.0, 1400.0 / max(h, w))
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    adaptive = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [gray, clahe, adaptive, otsu]


def run_ocr(reader, image: np.ndarray) -> tuple[str, np.ndarray]:
    variants = make_preprocessing_variants(image)
    best_text = ""
    best_image = variants[0]
    for variant in variants:
        lines = reader.readtext(variant, detail=0, paragraph=False)
        text = "\n".join(str(line).strip() for line in lines if str(line).strip())
        # Prefer useful text length, but don't reward repeated whitespace.
        score = len(re.sub(r"\s+", "", text))
        if score > len(re.sub(r"\s+", "", best_text)):
            best_text, best_image = text, variant
    if not best_text.strip():
        raise ValueError("No readable text was detected. Try a clearer, well-lit image.")
    return best_text, best_image


def detect_and_save_face(image: np.ndarray, record_id: str) -> Optional[Path]:
    """Best-effort face crop using available OpenCV Haar cascades.

    Tries original, contrast-enhanced, and upscaled grayscale images, plus both
    frontal-face and profile cascades when the installed OpenCV package includes them.
    Returns None safely if the OpenCV build lacks cascade support or no face is found.
    """
    try:
        if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return None

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        # Printed ID portraits can be small; enlarge at least 2x before Haar detection.
        factor = min(4.0, max(2.0, 1000.0 / max(h, w)))
        enlarged = cv2.resize(clahe, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC) if factor > 1 else clahe

        cascade_dir = Path(cv2.data.haarcascades)
        cascade_names = ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml", "haarcascade_profileface.xml"]
        candidates = []
        for name in cascade_names:
            cascade_file = cascade_dir / name
            if not cascade_file.exists():
                continue
            detector = cv2.CascadeClassifier(str(cascade_file))
            if detector.empty():
                continue
            for source, scale in ((gray, 1.0), (clahe, 1.0), (enlarged, factor)):
                try:
                    found = detector.detectMultiScale(
                        source, scaleFactor=1.08, minNeighbors=3,
                        minSize=(max(18, int(24 * scale)), max(18, int(24 * scale))),
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                except Exception:
                    continue
                for x, y, fw, fh in found:
                    if scale != 1.0:
                        x, y, fw, fh = int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)
                    # Filter tiny/noisy detections and very implausible aspect ratios.
                    ratio = fw / max(fh, 1)
                    if fw >= 24 and fh >= 24 and 0.55 <= ratio <= 1.5:
                        candidates.append((int(fw * fh), int(x), int(y), int(fw), int(fh)))

        if not candidates:
            return None
        _, x, y, fw, fh = max(candidates, key=lambda item: item[0])
        pad_x, pad_y = int(0.22 * fw), int(0.25 * fh)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)
        crop = image[y1:y2, x1:x2] if image.ndim == 3 else cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        if crop.size == 0:
            return None
        target = FACES_DIR / f"{record_id}.jpg"
        if cv2.imwrite(str(target), crop):
            return target
    except Exception:
        # Face detection is optional and must never cause the ID scan to fail.
        return None
    return None

def clean_ocr_line(line: str) -> str:
    line = line.replace("|", "I")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip(" \t:;,'\"")


def extract_fields(text: str) -> dict:
    """
    Heuristic extraction for common student/employee ID cards.
    OCR can misread characters, so extracted values should be verified by a human.
    """
    lines = [clean_ocr_line(x) for x in text.splitlines() if clean_ocr_line(x)]
    joined = "\n".join(lines)

    full_name = extract_name(lines)
    document_number = extract_document_number(lines, joined)
    date_of_birth = extract_labeled_date(lines, ("date of birth", "dob", "birth date", "born"))
    issue_date = extract_labeled_date(lines, ("issued on", "issue date", "date of issue", "issued"))
    expiry_date = extract_labeled_date(lines, ("valid upto", "valid up to", "valid uplo", "valid upo", "expiry date", "expires", "valid until", "expiration"))

    # Second pass: tolerate common OCR label errors such as "Issied" and "Uplo".
    # Only labeled dates are assigned, so unrelated dates aren't mistaken for expiry.
    date_fields = extract_dates(joined)
    date_of_birth = date_of_birth or date_fields.get("date_of_birth")
    issue_date = issue_date or date_fields.get("issue_date")
    expiry_date = expiry_date or date_fields.get("expiry_date")

    course = extract_course(lines)
    blood_group = extract_blood_group(lines)
    emergency_number = extract_emergency_number(lines)

    return {
        "full_name": full_name,
        "document_number": document_number,
        "date_of_birth": date_of_birth,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "course": course,
        "blood_group": blood_group,
        "emergency_number": emergency_number,
    }


def extract_name(lines: list[str]) -> Optional[str]:
    # First use explicit name labels.
    for i, line in enumerate(lines):
        m = re.search(r"(?:full\s*)?name\s*[:\-]\s*(.+)$", line, re.I)
        if m:
            candidate = normalize_name(m.group(1))
            if is_plausible_name(candidate):
                return candidate
        if re.fullmatch(r"(?:full\s*)?name", line, re.I) and i + 1 < len(lines):
            candidate = normalize_name(lines[i + 1])
            if is_plausible_name(candidate):
                return candidate

    # Card-specific heuristic: names are often a 2-4 word line, not a heading.
    excluded = (
        "faculty", "engineering", "technology", "mit-wpu", "mit wpu", "university",
        "blood group", "emergency", "issued", "valid", "course", "department",
        "student", "identity", "card", "mca", "bca", "b.tech", "m.tech",
    )
    candidates = []
    for line in lines:
        candidate = normalize_name(line)
        words = candidate.split()
        if not (2 <= len(words) <= 4) or not is_plausible_name(candidate):
            continue
        low = candidate.lower()
        if any(term in low for term in excluded):
            continue
        if any(char.isdigit() for char in candidate):
            continue
        # Names commonly contain title-case words; allow all-caps OCR too.
        candidates.append(candidate)
    # Prefer a line with 2-3 words and fewer characters, avoiding long headings.
    if candidates:
        candidates.sort(key=lambda value: (abs(len(value.split()) - 3), len(value)))
        return candidates[0]
    return None


def normalize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z .'-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .-'")
    return " ".join(word.capitalize() for word in value.split())


def is_plausible_name(value: str) -> bool:
    words = value.split()
    return bool(words) and all(len(re.sub(r"[^A-Za-z]", "", w)) >= 1 for w in words) and len(value) <= 60


def extract_document_number(lines: list[str], text: str) -> Optional[str]:
    patterns = [
        r"(?:id\s*(?:number|no\.?)|document\s*number|employee\s*id|card\s*no\.?|student\s*id)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-/]{4,})",
        r"\b(\d{8,14})\b",
    ]
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match:
                value = match.group(1).strip()
                if not re.fullmatch(r"\d{1,4}", value):
                    return value
    # Barcode numbers often appear on their own line without a label.
    for line in lines:
        value = re.sub(r"[\s|]", "", line)
        if re.fullmatch(r"\d{8,14}", value):
            return value
    return None


# Supports common printed formats including 07/2025, 06/2027, 07/2025,
# 07/12/2025, 7-12-25, and ISO dates. Month/year values are preserved as printed.
DATE_PATTERN = r"(?<!\d)(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{4})(?!\d)"


def _date_from_line(line: str) -> Optional[str]:
    """Find a date and tolerate OCR confusion around separators and labels."""
    normalized = line.replace("\\", "/").replace("|", "/")
    # First try standard numeric date formats.
    match = re.search(DATE_PATTERN, normalized)
    if match:
        return match.group(0).replace("-", "/") if re.fullmatch(r"\d{1,2}-\d{4}", match.group(0)) else match.group(0)

    # OCR sometimes reads a slash as a space or punctuation, e.g. 07 2025.
    match = re.search(r"(?<!\d)(0?[1-9]|1[0-2])\s*[.,' ]\s*(20\d{2})(?!\d)", normalized)
    if match:
        return f"{int(match.group(1)):02d}/{match.group(2)}"
    return None


def extract_labeled_date(lines: list[str], labels: tuple[str, ...]) -> Optional[str]:
    """Extract a date near its label without assigning unlabeled dates by guesswork."""
    normalized_labels = tuple(re.sub(r"[^a-z0-9]", "", label.lower()) for label in labels)
    for i, line in enumerate(lines):
        low = re.sub(r"[^a-z0-9]", "", line.lower())
        if any(label in low for label in normalized_labels):
            # Date may appear on the same OCR line or in the next line.
            found = _date_from_line(line)
            if found:
                return found
            if i + 1 < len(lines):
                found = _date_from_line(lines[i + 1])
                if found:
                    return found
    return None

def extract_course(lines: list[str]) -> Optional[str]:
    course_patterns = [
        (r"\bM\s*\.?\s*C\s*\.?\s*A\b", "MCA"),
        (r"\bM\s*\.?\s*B\s*\.?\s*A\b", "MBA"),
        (r"\bB\s*\.?\s*C\s*\.?\s*A\b", "BCA"),
        (r"\bB\s*\.?\s*\.?\s*TECH\b", "B.Tech"),
        (r"\bM\s*\.?\s*\.?\s*TECH\b", "M.Tech"),
        (r"\bB\s*\.?\s*\.?\s*E\b", "B.E."),
        (r"\bM\s*\.?\s*\.?\s*E\b", "M.E."),
        (r"\bP\s*\.?\s*H\s*\.?\s*D\b", "PhD"),
    ]
    for line in lines:
        for pattern, course in course_patterns:
            if re.search(pattern, line, re.I):
                return course
    return None


def extract_blood_group(lines: list[str]) -> Optional[str]:
    for line in lines:
        if "blood" in line.lower():
            match = re.search(r"\b(AB|A|B|O)\s*([+-])(?=\W|$)", line, re.I)
            if match:
                return re.sub(r"\s+", "", match.group(0)).upper()
    return None


def extract_emergency_number(lines: list[str]) -> Optional[str]:
    for line in lines:
        if "emergency" in line.lower():
            match = re.search(r"(?:\+?91[\s-]?)?[6-9]\d{9}\b", line)
            if match:
                return re.sub(r"\D", "", match.group(0))[-10:]
    return None

import re


def extract_dates(ocr_text):
    text = ocr_text.upper()

    # Normalize common OCR mistakes
    text = text.replace("ISSIED", "ISSUED")
    text = text.replace("ISSUEO", "ISSUED")
    text = text.replace("ISSUEDON", "ISSUED ON")
    text = text.replace("UPLO", "UPTO")
    text = text.replace("UPT0", "UPTO")
    text = text.replace("VALIDUNTIL", "VALID UNTIL")
    text = text.replace("EXPIRYDATE", "EXPIRY DATE")
    text = text.replace("DATEOFBIRTH", "DATE OF BIRTH")
    text = text.replace("VALID UPT0", "VALID UPTO")

    date_pattern = r"\b(0?[1-9]|1[0-2])\s*[/.-]\s*((?:19|20)?\d{2})\b"

    issue_date = None
    expiry_date = None
    date_of_birth = None

    issue_match = re.search(
        r"(?:ISSUED|ISSUE\s*DATE|ISSUED\s*ON)\D{0,20}"
        + date_pattern,
        text
    )

    expiry_match = re.search(
        r"(?:VALID\s*(?:UPTO|UP\s*TO|UNTIL)|EXPIRY(?:\s*DATE)?|EXPIRES)\D{0,30}"
        + date_pattern,
        text
    )
    dob_match = re.search(
        r"(?:DATE\s*OF\s*BIRTH|D\.?O\.?B\.?|BIRTH\s*DATE)\D{0,25}"
        + date_pattern,
        text
    )

    if issue_match:
        issue_date = issue_match.group(1) + "/" + issue_match.group(2)

    if expiry_match:
        expiry_date = expiry_match.group(1) + "/" + expiry_match.group(2)
    if dob_match:
        date_of_birth = dob_match.group(1) + "/" + dob_match.group(2)

    return {
        "date_of_birth": date_of_birth,
        "issue_date": issue_date,
        "expiry_date": expiry_date
    }