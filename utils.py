import copy, json, os, re
from functools import lru_cache
from pathlib import Path
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from PyPDF2 import PdfReader
from transformers import AutoTokenizer

OUTPUT_PATH = "/tmp/output_ocr"

#OUTPUT_PATH = "/home/peyzaguirre/notebooks/iarag/services/chunking/output/"
def detect_language(file):
    try:
        is_file = os.path.isfile(file)
    except (OSError, TypeError):
        is_file = False

    if isinstance(file, str) and not is_file:
        text = file
    else:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages[:1]:
            text += page.extract_text() or ""

    sample = " ".join(text.split()[:50])
    try:
        lang = detect(sample) if len(sample) > 0 else ""
    except LangDetectException:
        lang = ""

    if lang.startswith("es"):
        return "es"
    else:
        return "en"
   
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the tokenizer used by the semantic encoder and downstream index."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def token_length(text: str):
    return len(
        get_tokenizer().encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def split_md_by_titles(
    markdown_text: str,
    include_title: bool = True,
    preserve_blank_lines: bool = False,
):
    """ 
     split_md_by_titles() is a lightweight, line-based Markdown parser:
    - It treats every trimmed line beginning with # as a heading, 
    strips all # characters, and does not preserve heading depth.
    - It groups that heading with the following non-heading lines until the next heading.
    - Consecutive headings without body text are concatenated into one title.
    - Text before the first heading becomes an introductory section whose title is " ".
    - Empty lines are removed from the returned section text.
    - Headings with no body content can disappear.
    - It returns two aligned lists: current_titles and cleaned_sections. Each section currently includes its title as the first line.
        
    
    """
    sections = []
    current_titles = []
    current_content = []
    buffer_intro = []
    
    lines = markdown_text.splitlines()
    aux_title = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if aux_title != "" and current_content: # if there were previously titles and content save it first
                #block = "\n".join(current_titles) + "\n" + "\n".join(current_content).strip()
                current_titles.append(aux_title)
                body = "\n".join(current_content).strip()
                block = aux_title + "\n" + body if include_title else body
                sections.append(block.strip())
                aux_title = ""
                current_content = []
            # else append the title found
            aux_title = aux_title + " " +  stripped.lstrip("#").strip()
            
        elif stripped == "": # empty lines -> skipp if there is previously titles but not content
            if aux_title != "" and not current_content:
                continue
            else:
                if aux_title != "":
                    current_content.append(line)
                else:
                    buffer_intro.append(line)
        else: # normal content
            if aux_title != "":
                current_content.append(line)
            else:
                buffer_intro.append(line)
                
    if aux_title != "" and current_content:  # append last section
        #block = "\n".join(current_titles) + "\n" + "\n".join(current_content).strip()
        current_titles.append(aux_title)
        body = "\n".join(current_content).strip()
        block = aux_title + "\n" + body if include_title else body
        sections.append(block.strip())
        
    if buffer_intro:     # append the intro or first paragraph even without a title
        if preserve_blank_lines:
            intro = "\n".join(buffer_intro).strip()
        else:
            intro = "\n".join(
                [ln for ln in buffer_intro if ln.strip() != ""]
            ).strip()
        if intro:
            sections.insert(0, intro)
            current_titles.insert(0, " ")
            
    cleaned_sections = []     # check for empty lines and remove them
    for sec in sections:
        if preserve_blank_lines:
            cleaned_sections.append(sec.strip())
        else:
            lines = [ln.rstrip() for ln in sec.splitlines() if ln.strip() != ""]
            cleaned_sections.append("\n".join(lines))

    return current_titles, cleaned_sections


_SPLIT_PATTERNS = (
    r"\n[ \t]*\n+",       # paragraphs
    r"\n",                 # lines
    r"(?<=[.!?])\s+",     # sentences
    r"[ \t]+",             # words
)


def _split_keep_separator(text: str, pattern: str):
    parts = re.split(f"({pattern})", text)
    return [
        parts[index] + (parts[index + 1] if index + 1 < len(parts) else "")
        for index in range(0, len(parts), 2)
        if parts[index] or index + 1 < len(parts)
    ]


def _token_fallback(text: str, max_tokens: int):
    chunks = []
    while text:
        low, high, split_at = 1, len(text), 0
        while low <= high:
            middle = (low + high) // 2
            if token_length(text[:middle]) <= max_tokens:
                split_at = middle
                low = middle + 1
            else:
                high = middle - 1

        if split_at == 0:
            raise ValueError("max_split_tokens is too small for this section.")
        chunks.append(text[:split_at])
        text = text[split_at:]
    return chunks


def split_section_by_tokens(body: str, max_tokens: int):
    """Split a section at natural boundaries while enforcing a token limit."""
    def fits(text):
        return token_length(text.strip()) <= max_tokens

    def split_recursively(text, level=0):
        if fits(text):
            return [text]
        if level == len(_SPLIT_PATTERNS):
            return _token_fallback(text, max_tokens)

        parts = _split_keep_separator(text, _SPLIT_PATTERNS[level])
        if len(parts) == 1:
            return split_recursively(text, level + 1)

        result = []
        for part in parts:
            if part:
                result.extend(split_recursively(part, level + 1))
        return result

    pieces = split_recursively(body.strip())
    chunks = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if current and not fits(candidate):
            chunks.append(current.strip())
            current = piece
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    return chunks

