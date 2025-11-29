"""
Data ingestion module for UAV RAG system.
Loads:
    - Engineering manuals (.txt, .pdf)
    - Telemetry logs (.csv)
Outputs:
    - List[dict] where dict = {"text": "...", "metadata": {...}}
"""

import os
from pathlib import Path
import csv
from typing import List, Dict, Any

from .config import paths

# pdfplumber is optional; handle gracefully
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[INFO] pdfplumber not installed — PDF manual support disabled.")


# -----------------------------------------------------------
# 1. MANUAL LOADING (TXT + optional PDF)
# -----------------------------------------------------------

def load_manual_pdfs() -> List[Dict[str, Any]]:
    """
    Loads manuals from /data/manuals/ as dicts:
        [{"text": "...", "metadata": {"source": filename}}, ...]
    Supports:
        - .txt
        - .pdf   (if pdfplumber installed)
    """
    manual_dir = paths.manuals_dir
    docs = []

    print(f"[INFO] Manual directory: {manual_dir}")

    # ---------------------
    # Load TXT manuals
    # ---------------------
    for txt_path in manual_dir.glob("*.txt"):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append({
                    "text": text,
                    "metadata": {"source": txt_path.name}
                })
            else:
                print(f"[WARN] TXT manual empty: {txt_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to read TXT manual {txt_path}: {e}")

    # ---------------------
    # Load PDF manuals
    # ---------------------
    if PDF_AVAILABLE:
        for pdf_path in manual_dir.glob("*.pdf"):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                    text = "\n".join(pages)
                    if text.strip():
                        docs.append({
                            "text": text,
                            "metadata": {"source": pdf_path.name}
                        })
                    else:
                        print(f"[WARN] PDF manual empty: {pdf_path.name}")
            except Exception as e:
                print(f"[ERROR] Failed to read PDF manual {pdf_path}: {e}")
    else:
        if list(manual_dir.glob("*.pdf")):
            print("[WARN] PDF files detected but pdfplumber not installed.")

    print(f"[INFO] Loaded {len(docs)} manual documents.\n")
    return docs


# -----------------------------------------------------------
# 2. TELEMETRY CSV LOADING
# -----------------------------------------------------------

def load_telemetry_files() -> List[Dict[str, Any]]:
    """
    Loads all telemetry CSVs from /data/logs/.
    Produces one RAG record per row, formatted as:
        {
          "text": "IMU AccX=..., AccY=..., GPS HDOP=..., etc.",
          "metadata": {"source": filename, "timestamp": maybe}
        }
    """
    logs_dir = paths.logs_dir
    telemetry_records = []

    print(f"[INFO] Telemetry logs directory: {logs_dir}")

    for csv_path in logs_dir.glob("*.csv"):
        print(f"[INFO] Reading telemetry CSV: {csv_path.name}")

        try:
            with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if not row:
                        continue

                    # Build a readable telemetry string for RAG
                    text_parts = []
                    timestamp = None

                    for key, value in row.items():
                        if key is None or value is None:
                            continue

                        key_clean = key.strip()
                        val_clean = str(value).strip()

                        if not val_clean:
                            continue

                        # Detect possible timestamp
                        if key_clean.lower() in ["timestamp", "time", "t"]:
                            timestamp = val_clean

                        text_parts.append(f"{key_clean}: {val_clean}")

                    if not text_parts:
                        continue

                    telemetry_records.append({
                        "text": ", ".join(text_parts),
                        "metadata": {
                            "source": csv_path.name,
                            "timestamp": timestamp or "unknown"
                        }
                    })

        except Exception as e:
            print(f"[ERROR] Failed to read CSV log {csv_path}: {e}")

    print(f"[INFO] Loaded telemetry records: {len(telemetry_records)}\n")
    return telemetry_records
