from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from epub2txt import epub2txt


AVERAGE_WORDS_PER_MINUTE = 150
WORDS_PER_SECOND = AVERAGE_WORDS_PER_MINUTE / 60
TARGET_DURATION_SECONDS = 10
MIN_DURATION_SECONDS = 8
MAX_DURATION_SECONDS = 12

TARGET_WORD_COUNT = int(TARGET_DURATION_SECONDS * WORDS_PER_SECOND)
MIN_WORD_COUNT = max(1, int(MIN_DURATION_SECONDS * WORDS_PER_SECOND))
MAX_WORD_COUNT = int(MAX_DURATION_SECONDS * WORDS_PER_SECOND)


def clean_text(raw_text: str) -> str:
	text = raw_text.replace("\ufeff", " ")
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def split_into_sentences(text: str) -> List[str]:
	sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"])")
	candidates = sentence_endings.split(text)
	sentences = []
	for chunk in candidates:
		candidate = chunk.strip()
		if candidate:
			sentences.append(candidate)
	return sentences


def make_chunk(words: Sequence[str]) -> Tuple[str, int, float]:
	words_list = list(words)
	text = " ".join(words_list).strip()
	word_count = len(words_list)
	duration = round(word_count / WORDS_PER_SECOND, 2)
	return text, word_count, duration


def chunk_sentences(sentences: Iterable[str]) -> List[Tuple[str, int, float]]:
	chunks: List[Tuple[str, int, float]] = []
	current_words: List[str] = []

	def finalize(force: bool = False) -> None:
		nonlocal current_words
		if not current_words:
			return
		if not force and len(current_words) < MIN_WORD_COUNT:
			return
		chunks.append(make_chunk(current_words))
		current_words = []

	for sentence in sentences:
		words = sentence.split()
		if not words:
			continue

		index = 0
		while index < len(words):
			remaining_capacity = MAX_WORD_COUNT - len(current_words)
			if remaining_capacity <= 0:
				finalize(force=True)
				continue

			take = min(len(words) - index, remaining_capacity)
			current_words.extend(words[index : index + take])
			index += take

			if len(current_words) >= TARGET_WORD_COUNT:
				finalize(force=True)

	if current_words:
		if len(current_words) < MIN_WORD_COUNT and chunks:
			last_text, _, _ = chunks[-1]
			combined_words = last_text.split() + current_words
			chunks[-1] = make_chunk(combined_words)
		else:
			finalize(force=True)

	return chunks


def write_csv(csv_path: Path, segments: Sequence[Tuple[str, int, float]]) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["segment_index", "text", "word_count", "estimated_duration_seconds"])
		for index, (text, word_count, duration) in enumerate(segments, start=1):
			writer.writerow([index, text, word_count, f"{duration:.2f}"])


def process_epub(epub_path: Path, output_dir: Path) -> Path:
	text = epub2txt(str(epub_path))
	cleaned = clean_text(text)
	sentences = split_into_sentences(cleaned)
	segments = chunk_sentences(sentences)
	output_path = output_dir / f"{epub_path.stem}.csv"
	write_csv(output_path, segments)
	return output_path


def main() -> None:
	input_dir = Path("epubInput")
	output_dir = Path("csvOutput")

	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory not found: {input_dir.resolve()}")

	epub_files = sorted(input_dir.glob("*.epub"))
	if not epub_files:
		print(f"No EPUB files found in {input_dir.resolve()}")
		return

	for epub_path in epub_files:
		output_path = process_epub(epub_path, output_dir)
		print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()
