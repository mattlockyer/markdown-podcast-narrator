"""
Markdown Parser Module for Podcast Narrator
Parses markdown and converts to speech-ready chunks for TTS processing.

Supported elements:
  - Headings (#, ##, ###)
  - Paragraphs (bold/italic stripped to plain text)
  - Bullet and ordered lists
  - Fenced code blocks
  - Tables (GFM-style)
  - Admonitions / callouts (!!!info, !!!warning, etc.)
  - Horizontal rules
"""

from markdown_it import MarkdownIt
from mdit_py_plugins.admon import admon_plugin
from typing import List, Dict, Any, Tuple


# Pause durations (ms) inserted as silence between audio chunks
PAUSE_CHAPTER  = 1200   # before H1
PAUSE_SECTION  = 800    # before H2
PAUSE_SUB      = 500    # before H3+
PAUSE_PARA     = 400    # between paragraphs / blocks
PAUSE_LIST_END = 300    # after a list finishes
PAUSE_HR       = 1000   # horizontal rule

# Max chars per section for neural TTS.
# Higher = fewer calls (faster) but risks OOM on MPS for large sections.
# 600 keeps sections safe for Apple Silicon GPU memory.
MAX_SECTION_CHARS = 600


class MarkdownParser:
    """Parse markdown and convert to speech chunks for TTS processing."""

    def __init__(self):
        self.md = (
            MarkdownIt("commonmark")
            .use(admon_plugin)
            .enable("table")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_to_speech_tokens(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown text into speech tokens.

        Token shapes:
          {"type": "heading",    "level": int, "text": str}
          {"type": "paragraph",  "text": str}
          {"type": "list_start", "count": int}
          {"type": "list_item",  "index": int, "text": str}
          {"type": "list_end"}
          {"type": "code",       "language": str, "content": str}
          {"type": "table",      "headers": [str], "rows": [[str]]}
          {"type": "admonition", "kind": str, "title": str, "text": str}
          {"type": "hr"}
        """
        tokens = []
        md_tokens = self.md.parse(markdown_text)

        i = 0
        in_list = False
        list_item_index = 0

        def _count_list_items(start_idx: int, open_type: str, close_type: str) -> int:
            count, depth, j = 0, 0, start_idx + 1
            while j < len(md_tokens):
                if md_tokens[j].type == open_type:
                    depth += 1
                elif md_tokens[j].type == close_type:
                    if depth == 0:
                        break
                    depth -= 1
                elif md_tokens[j].type == "list_item_open" and depth == 0:
                    count += 1
                j += 1
            return count

        while i < len(md_tokens):
            token = md_tokens[i]

            if token.type == "heading_open":
                level = int(token.tag[1])
                i += 1
                if i < len(md_tokens) and md_tokens[i].type == "inline":
                    text = self._extract_text(md_tokens[i])
                    tokens.append({"type": "heading", "level": level, "text": text})
                i += 1

            elif token.type in ("bullet_list_open", "ordered_list_open"):
                in_list = True
                list_item_index = 0
                close_type = token.type.replace("open", "close")
                tokens.append({"type": "list_start", "count": _count_list_items(i, token.type, close_type)})

            elif token.type in ("bullet_list_close", "ordered_list_close"):
                in_list = False
                list_item_index = 0
                tokens.append({"type": "list_end"})

            elif token.type == "list_item_open" and in_list:
                list_item_index += 1
                j = i + 1
                item_text = ""
                while j < len(md_tokens) and md_tokens[j].type != "list_item_close":
                    if md_tokens[j].type == "inline":
                        item_text = self._extract_text(md_tokens[j])
                    j += 1
                tokens.append({"type": "list_item", "index": list_item_index, "text": item_text})
                i = j

            elif token.type == "paragraph_open" and not in_list:
                i += 1
                if i < len(md_tokens) and md_tokens[i].type == "inline":
                    text = self._extract_text(md_tokens[i])
                    tokens.append({"type": "paragraph", "text": text})
                i += 1

            elif token.type == "fence":
                tokens.append({"type": "code", "language": (token.info or "").strip(), "content": token.content.strip()})

            elif token.type == "code_block":
                tokens.append({"type": "code", "language": "", "content": token.content.strip()})

            elif token.type == "table_open":
                headers, rows, skip = self._parse_table(md_tokens, i)
                tokens.append({"type": "table", "headers": headers, "rows": rows})
                i = skip

            elif token.type == "admonition_open":
                kind  = (token.meta.get("tag") or "note").lower()
                title = token.content.strip()
                text_parts, past_title = [], False
                j = i + 1
                while j < len(md_tokens) and md_tokens[j].type != "admonition_close":
                    t = md_tokens[j]
                    if t.type == "admonition_title_close":
                        past_title = True
                    elif t.type == "inline" and past_title:
                        text_parts.append(self._extract_text(t))
                    j += 1
                tokens.append({"type": "admonition", "kind": kind, "title": title, "text": " ".join(text_parts)})
                i = j

            elif token.type == "hr":
                tokens.append({"type": "hr"})

            i += 1

        return tokens

    def tokens_to_speech_chunks(self, tokens: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Convert speech tokens to a list of (text, pause_after_ms) chunks.

        Each chunk is a natural speech segment that should be synthesized
        independently, producing better TTS quality than one giant string.
        The pause_after_ms value indicates how much silence to insert after
        the chunk's audio.
        """
        chunks: List[Tuple[str, int]] = []

        ORDINALS = {
            1: "First",  2: "Second", 3: "Third",  4: "Fourth", 5: "Fifth",
            6: "Sixth",  7: "Seventh",8: "Eighth", 9: "Ninth",  10: "Tenth",
        }

        ADMONITION_INTROS = {
            "info":     "Here's something worth knowing.",
            "note":     "Take note of this.",
            "tip":      "Here's a helpful tip.",
            "warning":  "An important warning.",
            "danger":   "Critical alert.",
            "success":  "Some good news.",
            "example":  "Let's look at an example.",
            "question": "A question to consider.",
        }

        h2_count = 0
        H2_INTROS = [
            "Now let's look at",
            "Moving on to",
            "Next up,",
            "Let's explore",
            "Turning our attention to",
        ]

        for idx, token in enumerate(tokens):
            t = token["type"]

            if t == "heading":
                level = token["level"]
                text  = token["text"]

                if level == 1:
                    chunks.append((f"{text}.", PAUSE_CHAPTER))
                elif level == 2:
                    intro = H2_INTROS[h2_count % len(H2_INTROS)]
                    h2_count += 1
                    chunks.append((f"{intro} {text}.", PAUSE_SECTION))
                else:
                    chunks.append((f"{text}.", PAUSE_SUB))

            elif t == "paragraph":
                chunks.append((token["text"], PAUSE_PARA))

            elif t == "list_start":
                count = token.get("count", 0)
                # Check if the *previous* chunk already introduces the list
                # (e.g. "Here are the tools you need:")
                prev_text = chunks[-1][0].lower() if chunks else ""
                already_introduced = any(w in prev_text for w in ["following", "here are", "these are", "you need", "you should", "to consider"])

                if not already_introduced:
                    if count == 1:
                        chunks.append(("There is one item to note.", 200))
                    elif count > 1:
                        chunks.append((f"Here are {count} key points.", 200))

            elif t == "list_item":
                ordinal = ORDINALS.get(token.get("index", 1), f"Item {token.get('index', 1)}")
                chunks.append((f"{ordinal}, {token['text']}.", 200))

            elif t == "list_end":
                # Just add extra pause after the list
                if chunks:
                    text, _ = chunks[-1]
                    chunks[-1] = (text, PAUSE_LIST_END)

            elif t == "code":
                lang    = token.get("language", "")
                content = token.get("content", "")

                if lang:
                    chunks.append((f"Here's a {lang} code example.", 300))
                else:
                    chunks.append(("Here's a code snippet.", 300))

                readable = _make_code_readable(content)
                if readable:
                    chunks.append((readable, PAUSE_PARA))

            elif t == "table":
                headers = token.get("headers", [])
                rows    = token.get("rows", [])

                col_names = ", ".join(headers) if headers else "unnamed columns"
                n_rows    = len(rows)
                row_word  = "row" if n_rows == 1 else "rows"

                chunks.append((
                    f"There is a table with columns: {col_names}, containing {n_rows} {row_word}.",
                    300,
                ))

                if rows and n_rows <= 5 and headers:
                    for row in rows:
                        cells = [f"{headers[ci]} is {cell}" for ci, cell in enumerate(row) if ci < len(headers) and cell.strip()]
                        chunks.append((f"{', '.join(cells)}.", 200))

                if chunks:
                    text, _ = chunks[-1]
                    chunks[-1] = (text, PAUSE_PARA)

            elif t == "admonition":
                kind  = token.get("kind", "note")
                title = token.get("title", "")
                text  = token.get("text", "")

                intro = ADMONITION_INTROS.get(kind, "Pay attention to this.")

                # Avoid "Important warning. Warning. ..." duplication
                if title.lower() == kind.lower() or not title:
                    body = text
                else:
                    body = f"{title}. {text}"

                chunks.append((f"{intro} {body}", PAUSE_PARA))

            elif t == "hr":
                if chunks:
                    text, _ = chunks[-1]
                    chunks[-1] = (text, PAUSE_HR)

        # Clean up: drop empty chunks, strip whitespace
        return [(text.strip(), pause) for text, pause in chunks if text.strip()]

    def tokens_to_section_chunks(self, tokens: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Group fine-grained chunks into section-level chunks capped at
        MAX_SECTION_CHARS (~500 chars).

        Splitting strategy:
          1. H1/H2 headings always start a new section.
          2. Within a section, if adding the next chunk would exceed the cap,
             flush the current section first.
          3. If a single chunk exceeds the cap, split it at sentence boundaries.

        This keeps each TTS call short enough for consistent emotion while
        giving the model enough context (100-500 chars) per call.
        """
        fine_chunks = self.tokens_to_speech_chunks(tokens)
        if not fine_chunks:
            return []

        sections: List[Tuple[str, int]] = []
        current_texts: List[str] = []
        current_len = 0
        current_pause = PAUSE_PARA

        def _flush():
            nonlocal current_texts, current_len
            if current_texts:
                sections.append((" ".join(current_texts), current_pause))
                current_texts = []
                current_len = 0

        for text, pause in fine_chunks:
            is_heading = pause >= PAUSE_SECTION

            # Heading always starts a new section
            if is_heading and current_texts:
                _flush()

            # Would this chunk push us over the limit?
            added_len = len(text) + (1 if current_texts else 0)
            if current_texts and current_len + added_len > MAX_SECTION_CHARS:
                _flush()

            # If a single chunk is itself too large, split at sentences
            if not current_texts and len(text) > MAX_SECTION_CHARS:
                for part in _split_at_sentences(text, MAX_SECTION_CHARS):
                    sections.append((part, PAUSE_PARA))
                # Override pause for last piece
                if sections:
                    t, _ = sections[-1]
                    sections[-1] = (t, pause)
                continue

            current_texts.append(text)
            current_len += added_len
            current_pause = pause

        _flush()
        return sections

    def tokens_to_speech_text(self, tokens: List[Dict[str, Any]]) -> str:
        """Convenience: flatten chunks into a single string (for previewing)."""
        chunks = self.tokens_to_speech_chunks(tokens)
        return "\n\n".join(text for text, _ in chunks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text(self, inline_token) -> str:
        """Extract plain text from an inline token, stripping all markup."""
        if not inline_token.children:
            return inline_token.content or ""

        parts = []
        for child in inline_token.children:
            if child.type == "text":
                parts.append(child.content)
            elif child.type == "softbreak":
                parts.append(" ")
            elif child.type == "code_inline":
                parts.append(child.content)
        return "".join(parts)

    def _parse_table(self, md_tokens, table_open_idx: int):
        headers: List[str] = []
        rows: List[List[str]] = []
        in_head = False
        in_body = False
        current_row: List[str] = []

        j = table_open_idx + 1
        while j < len(md_tokens):
            tok = md_tokens[j]
            if tok.type == "table_close":
                break
            elif tok.type == "thead_open":
                in_head = True
            elif tok.type == "thead_close":
                in_head = False
            elif tok.type == "tbody_open":
                in_body = True
            elif tok.type == "tbody_close":
                in_body = False
            elif tok.type == "tr_open":
                current_row = []
            elif tok.type == "tr_close":
                if in_head:
                    headers = current_row
                elif in_body:
                    rows.append(current_row)
                current_row = []
            elif tok.type in ("th_open", "td_open"):
                j += 1
                if j < len(md_tokens) and md_tokens[j].type == "inline":
                    current_row.append(self._extract_text(md_tokens[j]))
                    j += 1
            j += 1

        return headers, rows, j


# ------------------------------------------------------------------
# Standalone helpers
# ------------------------------------------------------------------

def _split_at_sentences(text: str, max_chars: int) -> List[str]:
    """Split text into pieces of at most max_chars, breaking at sentence ends."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and current_len + added > max_chars:
            parts.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += added

    if current:
        parts.append(" ".join(current))

    return parts if parts else [text]


def _make_code_readable(code: str) -> str:
    """
    Convert code content to a speakable summary.

    Rather than aggressively replacing every symbol (which produces
    unnatural results like "import times"), we just read lines with
    light cleanup and natural pausing.
    """
    lines = [l.rstrip() for l in code.splitlines() if l.strip()]
    if not lines:
        return ""

    if len(lines) > 12:
        return f"This is a {len(lines)}-line code block. It begins with: {lines[0].strip()}"

    # Light cleanup — only replace things that are truly unspeakable
    cleaned = []
    for line in lines:
        line = line.strip()
        line = line.replace("!=", " not equal ")
        line = line.replace("==", " equals ")
        line = line.replace(">=", " greater or equal ")
        line = line.replace("<=", " less or equal ")
        line = line.replace("->", " returns ")
        line = line.replace("=>", " maps to ")
        # Leave everything else as-is — TTS handles most syntax fine
        cleaned.append(line)

    return ". ".join(cleaned)


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    test_md = """
# Getting Started

Welcome to this guide on Python.

## Installation

Here are 3 steps to install:

- Download the installer from python.org
- Run the installer and follow prompts
- Verify with `python --version`

## Code Example

```python
def greet(name):
    return f"Hello, {name}!"
```

## Performance Data

| Service | Latency | Status |
|---------|---------|--------|
| API     | 120ms   | OK     |
| DB      | 80ms    | OK     |

!!! info "Pro Tip"
    Always use virtual environments for Python projects.

!!! warning
    Do not run as root in production.

---

### Summary

That wraps up the guide.
"""
    parser = MarkdownParser()
    tokens = parser.parse_to_speech_tokens(test_md)
    print("=== TOKENS ===")
    for tok in tokens:
        print(f"  {tok}")

    print("\n=== FINE CHUNKS ===")
    chunks = parser.tokens_to_speech_chunks(tokens)
    for text, pause in chunks:
        print(f"  [{pause:4d}ms] {text}")
    print(f"  Total: {len(chunks)} chunks")

    print("\n=== SECTION CHUNKS (for neural TTS) ===")
    sections = parser.tokens_to_section_chunks(tokens)
    for i, (text, pause) in enumerate(sections, 1):
        print(f"\n  Section {i} [{pause}ms pause after]:")
        print(f"    {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"    ({len(text)} chars)")
    print(f"\n  Total: {len(sections)} sections")
