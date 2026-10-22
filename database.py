"""SQLite persistence layer for Smart ID Card Scanner."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "smart_id_scanner.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                full_name TEXT,
                document_number TEXT,
                date_of_birth TEXT,
                issue_date TEXT,
                expiry_date TEXT,
                course TEXT,
                blood_group TEXT,
                emergency_number TEXT,
                metadata_json TEXT NOT NULL,
                face_image TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_document_number ON records(document_number)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_created_at ON records(created_at)")


def save_record(record: dict) -> None:
    metadata_json = json.dumps(record.get("metadata", {}), ensure_ascii=False)
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO records (
                record_id, full_name, document_number, date_of_birth, issue_date,
                expiry_date, course, blood_group, emergency_number,
                metadata_json, face_image
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["record_id"], record.get("full_name"), record.get("document_number"),
            record.get("date_of_birth"), record.get("issue_date"), record.get("expiry_date"),
            record.get("course"), record.get("blood_group"), record.get("emergency_number"),
            metadata_json, record.get("face_image")
        ))


def row_to_record(row: sqlite3.Row) -> dict:
    return {
        "record_id": row["record_id"],
        "full_name": row["full_name"],
        "document_number": row["document_number"],
        "date_of_birth": row["date_of_birth"],
        "issue_date": row["issue_date"],
        "expiry_date": row["expiry_date"],
        "course": row["course"],
        "blood_group": row["blood_group"],
        "emergency_number": row["emergency_number"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "face_image": row["face_image"],
        "created_at": row["created_at"],
    }


def get_record(record_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
    return row_to_record(row) if row else None


def list_records(limit: int = 20, offset: int = 0) -> tuple[list[dict], int]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM records ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [row_to_record(row) for row in rows], int(total)
