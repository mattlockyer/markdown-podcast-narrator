#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "InquirerPy>=0.3.4",
#     "markdown-it-py>=3.0.0",
#     "mdit-py-plugins>=0.4.0",
#     "click>=8.1.0",
#     "soundfile>=0.12.0",
#     "mlx-audio>=0.4.4",
#     "misaki[en]",
# ]
# ///
"""Interactive CLI for Markdown Podcast Narrator.

Provides a file-picker prompt so you can browse and select
a Markdown file, then narrates it to MP3.
"""

import os
import sys
from pathlib import Path

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from InquirerPy import inquirer

from parser import MarkdownParser
from narrator import Narrator, resolve_profile

# Remember where the user ran the command from
INVOKE_DIR = Path(os.getcwd()).resolve()

PARENT_DIR = "📁 .."


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Existing env vars take precedence (so shell overrides win).
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file(Path(__file__).resolve().parent / ".env")


def _list_entries(directory: Path) -> list[str]:
    """List subdirectories and .md files in a directory, sorted."""
    entries = [PARENT_DIR]
    dirs = []
    files = []
    try:
        for item in sorted(directory.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                dirs.append(f"📁 {item.name}/")
            elif item.suffix.lower() == ".md":
                files.append(f"📄 {item.name}")
    except PermissionError:
        pass
    return entries + dirs + files


def pick_markdown_file(start_dir: Path | None = None) -> Path:
    """Interactive directory browser. Arrow keys + Enter to navigate."""
    current = (start_dir or Path.home()).expanduser().resolve()

    while True:
        entries = _list_entries(current)
        choice = inquirer.select(
            message=f"  {current}",
            choices=entries,
            default=entries[1] if len(entries) > 1 else entries[0],
        ).execute()

        if choice == PARENT_DIR:
            current = current.parent
        elif choice.startswith("📁 "):
            dirname = choice[2:].rstrip("/").strip()
            current = current / dirname
        else:
            filename = choice[2:].strip()
            return (current / filename).resolve()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Interactive markdown narrator")
    ap.add_argument("-o", "--output", default=None,
                    help="Output audio file path (default: <input>.mp3)")
    ap.add_argument("--profile", default=None,
                    help="Voice profile: kokoro-transatlantic | chatterbox-transatlantic "
                         "(default: env MDPOD_PROFILE, else chatterbox-transatlantic)")
    ap.add_argument("--speaker", default=None,
                    help="Qwen3-TTS speaker name")
    ap.add_argument("--rate", default=None, type=float,
                    help="Speech rate multiplier 0.5-2.0")
    ap.add_argument("--fallback", action="store_true",
                    help="Use macOS 'say' instead of neural TTS")
    ap.add_argument("--engine", default=None,
                    choices=["qwen", "kokoro", "chatterbox", "macos"],
                    help="TTS engine (overrides the profile's engine)")
    ap.add_argument("--model", default=None,
                    help="Qwen3-TTS model ID")
    ap.add_argument("--instruct", default=None,
                    help="Narrator style instruction")
    ap.add_argument("--kokoro-voice", default=None,
                    help=("Kokoro voice spec. Single name (af_heart), preset "
                          "(narrator|transatlantic|professional|british|us), "
                          "equal blend (af_heart,bf_emma), or weighted blend "
                          "(af_heart:0.7+bf_emma:0.3)."))
    ap.add_argument("--chatterbox-model", default=None,
                    help="Chatterbox MLX repo (default: mlx-community/chatterbox-fp16)")
    ap.add_argument("--ref-audio", default=None,
                    help="Reference WAV for Chatterbox voice cloning")
    ap.add_argument("--exaggeration", default=None, type=float,
                    help="Chatterbox expressiveness 0-1")
    ap.add_argument("--cfg-weight", default=None, type=float,
                    help="Chatterbox CFG weight")
    ap.add_argument("--pause-scale", default=None, type=float,
                    help="Scale silence between sections (default per profile)")
    args = ap.parse_args()

    home_env = os.environ.get("MDPOD_HOME_PATH")
    start_dir = Path(home_env) if home_env else None
    selected_path = pick_markdown_file(start_dir)

    if args.output:
        output_file = Path(args.output)
        if not output_file.is_absolute():
            output_file = INVOKE_DIR / output_file
    else:
        output_file = INVOKE_DIR / selected_path.with_suffix(".mp3").name

    print(f"\nInput:  {selected_path}")
    print(f"Output: {output_file}\n")

    # Parse
    print("Parsing markdown...")
    content = selected_path.read_text(encoding="utf-8")
    parser = MarkdownParser()
    tokens = parser.parse_to_speech_tokens(content)
    if not tokens:
        print("Error: no content found in markdown file", file=sys.stderr)
        sys.exit(1)

    # Resolve voice profile (explicit flags override profile values)
    cfg = resolve_profile(
        args.profile, engine=args.engine, rate=args.rate, kokoro_voice=args.kokoro_voice,
        chatterbox_repo=args.chatterbox_model, chatterbox_ref_audio=args.ref_audio,
        chatterbox_exaggeration=args.exaggeration, chatterbox_cfg_weight=args.cfg_weight,
        pause_scale=args.pause_scale, instruct=args.instruct, speaker=args.speaker,
        model=args.model,
    )

    # Init TTS
    engine = "macos" if args.fallback else cfg["engine"]
    print(f"Initializing TTS ({engine}, profile={cfg['profile']})...")
    narrator = Narrator(engine=engine, model_id=cfg["model"],
                        chatterbox_repo=cfg["chatterbox_repo"],
                        chatterbox_ref_audio=cfg["chatterbox_ref_audio"],
                        chatterbox_exaggeration=cfg["chatterbox_exaggeration"],
                        chatterbox_cfg_weight=cfg["chatterbox_cfg_weight"],
                        pause_scale=cfg["pause_scale"])

    if not narrator.initialize():
        if engine != "macos":
            print(f"{engine} unavailable, falling back to macOS 'say'...")
            narrator = Narrator(engine="macos")
            if not narrator.initialize():
                print("Error: no TTS backend available", file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: TTS initialization failed", file=sys.stderr)
            sys.exit(1)

    narrator.set_voice_params(rate=cfg["rate"], speaker=cfg["speaker"],
                              instruct=cfg["instruct"], kokoro_voice=cfg["kokoro_voice"])

    # Choose chunk strategy
    if narrator.is_neural:
        sections = parser.tokens_to_section_chunks(tokens)
        if not sections:
            print("Error: no speakable content", file=sys.stderr)
            sys.exit(1)

        total = sum(len(t) for t, _ in sections)
        print(f"Prepared {len(sections)} sections ({total} chars)")
        print("Generating audio (section-by-section)...")

        def on_progress(current, total):
            print(f"  Section {current}/{total}...", end="\r")

        ok = narrator.synthesize_sections(sections, str(output_file), on_progress)
        print()
    else:
        chunks = parser.tokens_to_speech_chunks(tokens)
        if not chunks:
            print("Error: no speakable content", file=sys.stderr)
            sys.exit(1)

        total = sum(len(t) for t, _ in chunks)
        print(f"Prepared {len(chunks)} chunks ({total} chars)")
        print("Generating audio...")
        ok = narrator.synthesize_chunks(chunks, str(output_file))

    if not ok:
        print("Error: audio generation failed", file=sys.stderr)
        sys.exit(1)

    size_kb = output_file.stat().st_size / 1024
    print(f"Saved to {output_file} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
