"""
Markdown Podcast Narrator

Converts a Markdown document into a podcast-style audio file
using Qwen3-TTS, Kokoro, or macOS 'say' as a fallback.

Strategy:
  - macOS 'say':
    Sends the entire document in a single call with [[slnc N]]
    embedded pause commands. This helps maintain consistent
    pacing and emotion across the full narration.

  - Qwen3-TTS:
    Uses section-level chunks (grouped by headings). Each section
    is large enough to give the neural model context for a stable
    narrator tone. Sections are stitched together with PCM silence
    to create natural pauses.

  - Kokoro:
    Uses medium-sized chunks (grouped by paragraphs or short
    sections) to keep generation fast while preserving natural
    phrasing. Audio segments are concatenated with short pauses
    so the narration flows smoothly. Kokoro is optimized for
    fast local inference while still producing natural speech.
"""

import sys
from pathlib import Path

import click

from parser import MarkdownParser
from narrator import Narrator, resolve_profile


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", default=None,
              help="Output audio file path (default: output.mp3)")
@click.option("--profile", default=None,
              help="Voice profile: kokoro-transatlantic | chatterbox-transatlantic. "
                   "Default: env MDPOD_PROFILE, else chatterbox-transatlantic")
@click.option("--speaker", default=None, help="Qwen3-TTS speaker name")
@click.option("--rate", default=None, type=float, help="Speech rate multiplier 0.5-2.0")
@click.option("--fallback", is_flag=True, help="Use macOS 'say' instead of neural TTS")
@click.option("--engine", default=None,
              type=click.Choice(["qwen", "kokoro", "chatterbox", "macos"]),
              help="TTS engine (overrides the profile's engine)")
@click.option("--model", default=None,
              help="Qwen3-TTS model ID (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)")
@click.option("--instruct", default=None,
              help="Narrator style instruction for Qwen3-TTS (e.g. 'Speak slowly and calmly')")
@click.option("--kokoro-voice", default=None,
              help=("Kokoro voice spec: single name (af_heart), preset "
                    "(narrator|transatlantic|professional|british|us), equal blend "
                    "(af_heart,bf_emma), or weighted (af_heart:0.7+bf_emma:0.3). "
                    "Default: narrator preset"))
@click.option("--chatterbox-model", default=None,
              help="Chatterbox MLX repo (default: mlx-community/chatterbox-fp16)")
@click.option("--ref-audio", default=None, type=click.Path(exists=True),
              help="Reference WAV for Chatterbox voice cloning (default: bundled voice)")
@click.option("--exaggeration", default=None, type=float,
              help="Chatterbox emotion/expressiveness 0-1 (default: 0.5)")
@click.option("--cfg-weight", default=None, type=float,
              help="Chatterbox classifier-free guidance weight (default: 0.5)")
@click.option("--pause-scale", default=None, type=float,
              help="Scale silence between sections/paragraphs (default: 1.0; "
                   "try ~0.4 for Chatterbox, which already self-pauses)")
def cli(input_file: str, output_file: str, profile: str, speaker: str, rate: float,
        fallback: bool, engine: str, model: str, instruct: str,
        kokoro_voice: str, chatterbox_model: str, ref_audio: str,
        exaggeration: float, cfg_weight: float, pause_scale: float):
    """Convert a Markdown file to podcast-style audio narration.

    INPUT_FILE: Path to the Markdown (.md) file to convert.
    """
    try:
        if not output_file:
            output_file = "output.mp3"

        output_path = Path(output_file)
        if output_path.suffix.lower() not in (".wav", ".mp3"):
            output_file = str(output_path.with_suffix(".mp3"))
            click.echo("Note: output format changed to .mp3")

        # --- Parse ---
        click.echo("Parsing markdown...")
        content = Path(input_file).read_text(encoding="utf-8")

        parser = MarkdownParser()
        tokens = parser.parse_to_speech_tokens(content)
        if not tokens:
            click.echo("Error: no content found in markdown file", err=True)
            sys.exit(1)

        # --- Resolve voice profile (explicit flags override profile values) ---
        cfg = resolve_profile(
            profile, engine=engine, rate=rate, kokoro_voice=kokoro_voice,
            chatterbox_repo=chatterbox_model, chatterbox_ref_audio=ref_audio,
            chatterbox_exaggeration=exaggeration, chatterbox_cfg_weight=cfg_weight,
            pause_scale=pause_scale, instruct=instruct, speaker=speaker, model=model,
        )
        engine = "macos" if fallback else cfg["engine"]

        # --- Init TTS ---
        click.echo(f"Initializing TTS ({engine}, profile={cfg['profile']})...")
        narrator = Narrator(engine=engine, model_id=cfg["model"],
                            chatterbox_repo=cfg["chatterbox_repo"],
                            chatterbox_ref_audio=cfg["chatterbox_ref_audio"],
                            chatterbox_exaggeration=cfg["chatterbox_exaggeration"],
                            chatterbox_cfg_weight=cfg["chatterbox_cfg_weight"],
                            pause_scale=cfg["pause_scale"])

        if not narrator.initialize():
            if engine != "macos":
                click.echo(f"{engine} unavailable, falling back to macOS 'say'...")
                narrator = Narrator(engine="macos")
                if not narrator.initialize():
                    click.echo("Error: no TTS backend available", err=True)
                    sys.exit(1)
            else:
                click.echo("Error: TTS initialization failed", err=True)
                sys.exit(1)

        narrator.set_voice_params(rate=cfg["rate"], speaker=cfg["speaker"],
                                  instruct=cfg["instruct"], kokoro_voice=cfg["kokoro_voice"])

        # --- Choose chunk strategy based on backend ---
        is_neural = narrator.is_neural

        if is_neural:
            # Section-level chunks for neural TTS — consistent emotion
            sections = parser.tokens_to_section_chunks(tokens)
            if not sections:
                click.echo("Error: no speakable content", err=True)
                sys.exit(1)

            total_chars = sum(len(t) for t, _ in sections)
            max_chars = max(len(t) for t, _ in sections)
            click.echo(f"Prepared {len(sections)} sections ({total_chars} chars, max {max_chars}/section)")
            click.echo("Generating audio (section-by-section)...")

            def on_progress(current: int, total: int):
                click.echo(f"  Section {current}/{total}...", nl=False)
                click.echo("\r", nl=False)

            ok = narrator.synthesize_sections(sections, output_file, on_progress)
            click.echo()
        else:
            # Fine-grained chunks for macOS 'say' — single call with [[slnc]]
            chunks = parser.tokens_to_speech_chunks(tokens)
            if not chunks:
                click.echo("Error: no speakable content", err=True)
                sys.exit(1)

            total_chars = sum(len(t) for t, _ in chunks)
            click.echo(f"Prepared {len(chunks)} chunks ({total_chars} chars)")
            click.echo("Generating audio (single narration)...")

            ok = narrator.synthesize_chunks(chunks, output_file)

        if not ok:
            click.echo("Error: audio generation failed", err=True)
            sys.exit(1)

        size_kb = Path(output_file).stat().st_size / 1024
        click.echo(f"Saved to {output_file} ({size_kb:.0f} KB)")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
