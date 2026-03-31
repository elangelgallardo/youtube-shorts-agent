"""Convert TTS timepoints into SRT subtitle files."""
import re
from pathlib import Path

from models.video_job import Scene, Timepoint


def generate_srt(
    scenes: list[Scene],
    timepoints: list[Timepoint],
    output_path: Path,
    words_per_chunk: int = 4,
) -> None:
    """
    Groups word-level timepoints into subtitle chunks of `words_per_chunk` words
    and writes an SRT file to `output_path`.
    """
    # Build a flat list of (mark_name, time_seconds) for word marks only
    word_tp_pattern = re.compile(r"^s(\d+)_w(\d+)$")
    scene_end_pattern = re.compile(r"^scene_(\d+)_end$")

    # Map scene_id -> end_time from scene_N_end marks
    scene_end_times: dict[int, float] = {}
    for tp in timepoints:
        m = scene_end_pattern.match(tp.mark_name)
        if m:
            scene_end_times[int(m.group(1))] = tp.time_seconds

    # Gather word timepoints per scene
    scene_words: dict[int, list[tuple[int, float, str]]] = {}  # scene_id -> [(word_idx, time, text)]
    for tp in timepoints:
        m = word_tp_pattern.match(tp.mark_name)
        if m:
            scene_id = int(m.group(1))
            word_idx = int(m.group(2))
            scene_words.setdefault(scene_id, []).append((word_idx, tp.time_seconds, ""))

    # Populate word text from scenes
    scene_map = {s.scene_id: s for s in scenes}
    for scene_id, entries in scene_words.items():
        if scene_id not in scene_map:
            continue
        words = _split_words(scene_map[scene_id].spoken_text)
        for i, (widx, t, _) in enumerate(entries):
            text = words[widx] if widx < len(words) else ""
            scene_words[scene_id][i] = (widx, t, text)

    # Build flat word list in order
    all_words: list[tuple[float, float, str]] = []  # (start, end, text)
    for scene_id in sorted(scene_words.keys()):
        entries = sorted(scene_words[scene_id], key=lambda x: x[0])
        end_time = scene_end_times.get(scene_id, entries[-1][1] + 2.0 if entries else 0.0)
        for j, (_, t_start, text) in enumerate(entries):
            if j + 1 < len(entries):
                t_end = entries[j + 1][1]
            else:
                t_end = end_time
            if text:
                all_words.append((t_start, t_end, text))

    if not all_words:
        # Fallback: line-based splits from scene narration
        _fallback_srt(scenes, output_path)
        return

    # Chunk into groups
    chunks = _chunk_words(all_words, words_per_chunk)

    # Write SRT
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(chunks, start=1):
            f.write(f"{idx}\n")
            f.write(f"{_fmt_time(start)} --> {_fmt_time(end)}\n")
            f.write(f"{text}\n\n")


def _chunk_words(
    words: list[tuple[float, float, str]], n: int
) -> list[tuple[float, float, str]]:
    chunks = []
    for i in range(0, len(words), n):
        group = words[i : i + n]
        start = group[0][0]
        end = group[-1][1]
        text = " ".join(w[2] for w in group)
        chunks.append((start, end, text))
    return chunks


def _fallback_srt(scenes: list[Scene], output_path: Path) -> None:
    """Simple equal-duration subtitle splits when timepoints are unavailable."""
    lines: list[tuple[float, float, str]] = []
    cursor = 0.0
    for scene in scenes:
        sentences = [s.strip() for s in scene.spoken_text.split(".") if s.strip()]
        if not sentences:
            continue
        dur = scene.duration_hint_s / max(len(sentences), 1)
        for sent in sentences:
            lines.append((cursor, cursor + dur, sent + "."))
            cursor += dur

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(lines, start=1):
            f.write(f"{idx}\n")
            f.write(f"{_fmt_time(start)} --> {_fmt_time(end)}\n")
            f.write(f"{text}\n\n")


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _split_words(text: str) -> list[str]:
    import re
    return [t for t in re.split(r"\s+", text.strip()) if t]
