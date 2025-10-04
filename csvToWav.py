from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Set

import requests
from dotenv import load_dotenv


CSV_DIR = Path("csvOutput")
WAV_DIR = Path("wavOutput")
PROGRESS_DIR = WAV_DIR / ".progress"


def ensure_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is not set. Please add it to your environment or .env file."
        )
    return api_key


def select_csv_file(csv_dir: Path) -> Path:
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {csv_dir.resolve()}. Generate one with epubToCsv.py first."
        )

    print("Available CSV files:")
    for idx, csv_file in enumerate(csv_files, start=1):
        print(f"  {idx}. {csv_file.name}")

    while True:
        choice = input("Select a file by number (or press Enter for 1): ").strip()
        if not choice:
            return csv_files[0]
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(csv_files):
                return csv_files[index - 1]
        print("Invalid selection. Please enter a valid number.")


def progress_path(csv_path: Path) -> Path:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return PROGRESS_DIR / f"{csv_path.stem}.json"


def load_completed_indices(progress_file: Path) -> Set[int]:
    if not progress_file.exists():
        return set()
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        indices = data.get("completed_indices", [])
        return {int(idx) for idx in indices}
    except (json.JSONDecodeError, ValueError):
        print(f"Warning: Progress file {progress_file} is corrupted. Starting fresh.")
        return set()


def save_completed_indices(progress_file: Path, indices: Iterable[int]) -> None:
    sorted_indices = sorted(set(indices))
    payload = {"completed_indices": sorted_indices}
    progress_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def synthesize_speech(
    text: str,
    output_path: Path,
    api_key: str,
    model: str = "aura-2-neptune-en",
    encoding: str = "linear16",
    container: str = "wav",
    timeout: float = 60.0,
) -> None:
    url = (
        "https://api.deepgram.com/v1/speak"
        f"?model={model}&encoding={encoding}&container={container}"
    )
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Deepgram synthesis failed ({response.status_code}): {response.text}"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_segment_index(row: dict, fallback: int) -> int:
    raw_index = row.get("segment_index") or row.get("index")
    if raw_index:
        try:
            return int(raw_index)
        except ValueError:
            pass
    return fallback


def convert_csv(csv_path: Path, wav_dir: Path, api_key: str) -> None:
    wav_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_path(csv_path)
    completed = load_completed_indices(progress_file)

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row.")
        if "text" not in reader.fieldnames:
            raise ValueError(
                f"CSV file {csv_path} must contain a 'text' column. Found: {reader.fieldnames}"
            )

        total_rows = 0
        converted_rows = 0

        try:
            for row_number, row in enumerate(reader, start=1):
                total_rows += 1
                segment_index = parse_segment_index(row, row_number)

                if segment_index in completed:
                    continue

                text = (row.get("text") or "").strip()
                if not text:
                    completed.add(segment_index)
                    save_completed_indices(progress_file, completed)
                    continue

                output_file = wav_dir / f"{csv_path.stem}_{segment_index:04d}.wav"
                if output_file.exists():
                    completed.add(segment_index)
                    save_completed_indices(progress_file, completed)
                    continue

                print(f"Generating audio for segment {segment_index}...")
                try:
                    synthesize_speech(text, output_file, api_key)
                except RuntimeError as exc:
                    print(f"Error on segment {segment_index}: {exc}")
                    print("Conversion halted. Fix the issue and rerun to resume.")
                    return

                completed.add(segment_index)
                converted_rows += 1
                save_completed_indices(progress_file, completed)
                print(f"Saved {display_path(output_file)}")

        except KeyboardInterrupt:
            print("\nConversion interrupted by user. Progress saved.")
            return

    print(
        f"Finished processing {csv_path.name}: {converted_rows} new segments converted, "
        f"{len(completed)} total segments marked complete."
    )


def main() -> None:
    if not CSV_DIR.exists():
        raise FileNotFoundError(
            f"CSV directory not found: {CSV_DIR.resolve()}. Create it or adjust CSV_DIR."
        )

    api_key = ensure_api_key()
    csv_path = select_csv_file(CSV_DIR)
    print(f"Selected {csv_path.name}\n")
    convert_csv(csv_path, WAV_DIR, api_key)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)