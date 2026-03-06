# AGENTS.md — Context and Guidelines for AI Agents

This file is the source of truth for any AI agent (Claude Code, Cursor, Copilot, etc.) working on this codebase. Read this before making changes.

---

## What this project is

A local Python CLI that converts Markdown documents into podcast-style audio. The user runs `uv run main.py input.md` and gets a `.wav` file.

**Primary user environment:**
- macOS, Apple Silicon M4 Max
- Python 3.12 via `uv`
- Multiple TTS engines: Qwen3-TTS (MPS), Kokoro (CPU), macOS `say` (fallback)

---

## File map and responsibilities

| File | Owns | Must not touch |
|---|---|---|
| `parser.py` | Markdown parsing, token generation, chunk + pause decisions | Audio, subprocess, file I/O |
| `narrator.py` | TTS backends, audio stitching, WAV writing | Markdown, chunk logic |
| `main.py` | CLI flags, pipeline orchestration, fallback logic | Parsing internals, audio internals |
| `pyproject.toml` | Dependency declarations for uv | Do not add heavy deps to core |
| `requirements.txt` | Flat dep list for pip users | Keep in sync with pyproject.toml |
| `example.md` | Test document covering all supported elements | Keep all element types present |

**One-sentence rule:** `parser.py` knows nothing about audio. `narrator.py` knows nothing about Markdown. `main.py` is the only place they meet.

---

## Core data contract

The interface between parser and narrator is a list of chunks:

```python
List[Tuple[str, int]]  # (text_to_speak, pause_after_ms)
```

- `parser.py` produces this via `tokens_to_speech_chunks()` (fine-grained) or `tokens_to_section_chunks()` (section-level)
- `narrator.py` consumes this via `synthesize_chunks()` (macOS say) or `synthesize_sections()` (neural engines)
- `main.py` passes it through unchanged, choosing which method based on `narrator.is_neural`

**Do not break this contract.** If you need to add metadata to chunks (e.g. speaking rate per chunk), extend to `List[Tuple[str, int, dict]]` and update both sides.

---

## Key architectural decisions

### Why chunk-based synthesis?
Sending large text to a TTS model in one call degrades output quality. Splitting at natural boundaries (headings, paragraphs, list items) keeps each TTS call focused, then silence frames are inserted at stitch points.

### Why section-level grouping for neural TTS?
Fine-grained chunks (per paragraph) produced inconsistent emotion between segments — each call started fresh. Grouping chunks under H1/H2 headings into sections (300-800 chars) gives the model enough context for consistent prosody within a section while keeping memory usage safe.

### `MAX_SECTION_CHARS` tradeoff
Defined in `parser.py`. Higher = fewer synthesis calls (faster) but risks GPU OOM on large sections and slight prosody drift. Lower = safer, more consistent, more calls. Default: 600. The right value depends on document length and available GPU memory.

### Why single-call for macOS `say` but section-level for neural TTS?
macOS `say` sends the ENTIRE document as one call with `[[slnc N]]` embedded pause commands. This preserves consistent prosody and emotion — no chunk boundaries means no mood shifts. Neural engines (Qwen3-TTS, Kokoro) use section-level chunks because they generate audio autoregressively — shorter inputs with consistent `instruct` give better quality than one giant input.

### Why incremental PCM writing?
`_synth_chunked` writes PCM bytes to a temp file section by section instead of accumulating in a list. For a 55KB document, accumulating all audio in RAM before writing can exhaust system memory. The temp file approach keeps peak memory bounded to one section at a time.

### Why `torch.inference_mode()`?
Wrapping Qwen3-TTS calls in `torch.inference_mode()` disables gradient tracking, reducing memory usage and slightly speeding up inference. The model doesn't need gradients during generation.

### Why `torch.mps.empty_cache()` between sections?
MPS (Metal) allocates GPU memory lazily and doesn't always release it immediately. Calling `empty_cache()` after each section prevents gradual accumulation that would eventually OOM the process.

### Why no pydub / ffmpeg?
To keep the dependency footprint minimal and avoid the ffmpeg binary requirement. WAV files are written directly using Python's `struct` module. This is sufficient for mono int16 PCM.

### Why stdin for macOS `say`?
The macOS `say` CLI has a ~256KB argument length limit. Passing long text as a CLI argument silently truncates. Using `subprocess.run(..., input=text)` sends text via stdin which has no such limit.

### Why is Qwen3-TTS optional in pyproject.toml?
`torch` is 2GB+. Users who only want macOS fallback or Kokoro should not need to download it.

### Why ALL-CAPS text preprocessing?
TTS engines often mispronounce acronyms (reading "API" as a word rather than "A P I"). The `_prepare_text_for_tts()` function in `narrator.py` handles this before every synthesis call, for all engines.

---

## TTS Engine backends

### Adding a new engine

1. Add `_init_<engine>(self) -> bool` to `Narrator` — load model, return True on success
2. Add `_synth_<engine>(self, text: str) -> Tuple[Optional[bytes], Optional[int]]` — return `(pcm_int16_bytes, sample_rate)`
3. Add engine name to `Narrator.ENGINES` tuple
4. Add dispatch case in `initialize()` and `_synth_one()`
5. Add `--engine` choice to `main.py`'s click option

**Required output format from any backend:** raw int16 PCM bytes at a consistent sample rate. The `_make_silence()` and `_write_wav()` functions handle everything else.

### Current engines

| Engine | Init method | Synth method | Backend |
|---|---|---|---|
| `qwen` | `_init_qwen` | `_synth_qwen` | Qwen3-TTS on MPS/CUDA/CPU |
| `kokoro` | `_init_kokoro` | `_synth_kokoro` | Kokoro-82M on CPU |
| `macos` | `_init_macos_say` | `_synth_single_macos` | macOS `say` subprocess |

---

## How to add a new Markdown element

1. In `parser.py`, handle the relevant markdown-it-py token types in `parse_to_speech_tokens()`
2. Add a new token dict shape to the docstring
3. Handle the new token type in `tokens_to_speech_chunks()`
4. Add the element to `example.md` to test it
5. Run `python parser.py` and verify the chunk output looks right

**Current token types:**
- `heading` (level: 1-6, text)
- `paragraph` (text)
- `list_start` (count)
- `list_item` (index, text)
- `list_end`
- `code` (language, content)
- `table` (headers, rows)
- `admonition` (kind, title, text)
- `hr`

---

## Platform-specific components

| Component | macOS | Linux | Windows |
|---|---|---|---|
| Neural TTS | Qwen3-TTS on MPS, Kokoro on CPU | Qwen3-TTS on CUDA, Kokoro on CPU | Qwen3-TTS on CUDA/CPU, Kokoro on CPU |
| Fallback TTS | `say` via subprocess | `espeak-ng` or `piper` | `pyttsx3` or SAPI |
| AIFF→WAV | `afconvert` (built-in) | Not needed | Not needed |
| sox | `brew install sox` | `apt install sox` | `choco install sox` |

If adding Linux/Windows support, only `narrator.py` needs a new fallback backend and `main.py` needs platform detection. The parser and WAV writer are already cross-platform.

---

## What NOT to do

- **Do not add ffmpeg as a hard dependency.** Use `afconvert` on macOS or optional imports for format conversion.
- **Do not merge parser and narrator logic.** Keep the data contract clean.
- **Do not send very large text (>1500 chars) to neural TTS in a single call.** Always respect `MAX_SECTION_CHARS`.
- **Do not use `sys.argv` directly in tests.** Use Click's `CliRunner` for testing `main.py`.
- **Do not hardcode paths.** All paths already use `pathlib.Path`.
- **Do not add docstrings or comments to unchanged code** when making targeted edits.
- **Do not break the `--fallback` flag.** It must always work without any model downloaded.

---

## Open extension points

1. **MP3 output** — add `--format mp3`, import `pydub` conditionally, convert the final WAV
2. **Voice cloning** — swap `generate_custom_voice` for `generate_voice_clone` in `_synth_qwen()`, expose `--ref-audio` and `--ref-text` flags
3. **Voice design** — use `VoiceDesign` model variant, expose `--voice-description` flag
4. **Per-heading voice change** — extend chunk tuple to include a voice hint
5. **Streaming output** — write WAV chunks to disk as they complete instead of waiting for all
6. **CosyVoice / other engines** — follow the engine addition steps above; `_synth_one()` dispatch is the only place to hook in

---

## Testing

No test suite yet. To manually verify:

```bash
# Test parser only (prints token list and chunk preview)
python parser.py

# Test full pipeline with zero model download
uv run main.py example.md --fallback

# Test Kokoro engine
uv run main.py example.md --engine kokoro -o /tmp/test_kokoro.wav

# Test Qwen3-TTS (requires qwen-tts + torch + sox)
uv run main.py example.md -o /tmp/test_qwen.wav
```

When adding new markdown elements, always update `example.md` to include an example, then run the parser test to confirm chunks look right before generating audio.

---

## Known limitations

- Only mono audio output (single channel). Stereo would require updating `_write_wav()` and all silence generation.
- Qwen3-TTS `CustomVoice` has a fixed set of speakers. "Ryan" and "Emily" are confirmed to work.
- Very long paragraphs (>600 chars after section splitting) may produce uneven TTS quality. The `_split_at_sentences()` helper handles this automatically.
- The admonition title deduplication (`if title.lower() == kind.lower()`) is a heuristic that fails for custom-titled admonitions that happen to match the type name.
- Code block symbol replacement is deliberately minimal. Some languages (APL, Perl) will still produce strange results.
- Kokoro outputs at 24000 Hz sample rate; Qwen3-TTS sample rate varies by model. Silence frames are generated at the correct rate per engine.
