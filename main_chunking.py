import copy, json, os
from pathlib import Path
from semantic_router.encoders import HuggingFaceEncoder
import torch
from semantic_router.splitters import RollingWindowSplitter
from utils import split_md_by_titles, split_section_by_tokens, token_length


DEFAULT_MODEL = os.getenv("MODEL_NAME", "intfloat/e5-large-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = HuggingFaceEncoder(
    name=DEFAULT_MODEL,
    device=device 
)


splitter_txt = RollingWindowSplitter(
    encoder=encoder,
    dynamic_threshold=True,
    min_split_tokens=100,
    max_split_tokens=200,
    split_tokens_tolerance=10,
    window_size=2,
    plot_splits=False, 
    enable_statistics=True  
)


def _join_docs(docs):
    return " ".join(doc.strip() for doc in docs if doc.strip()).strip()


def _rebalance_small_tail(split_docs, min_split_tokens, max_split_tokens):
    """Merge or rebalance an undersized final split within one title section."""
    if len(split_docs) < 2:
        return split_docs

    last_text = _join_docs(split_docs[-1])
    if token_length(last_text) >= min_split_tokens:
        return split_docs

    combined_docs = split_docs[-2] + split_docs[-1]
    combined_text = _join_docs(combined_docs)
    if token_length(combined_text) <= max_split_tokens:
        return split_docs[:-2] + [combined_docs]

    best_split = None
    best_score = None
    for index in range(1, len(combined_docs)):
        left_docs = combined_docs[:index]
        right_docs = combined_docs[index:]
        left_tokens = token_length(_join_docs(left_docs))
        right_tokens = token_length(_join_docs(right_docs))
        if left_tokens > max_split_tokens or right_tokens > max_split_tokens:
            continue

        deficit = max(0, min_split_tokens - left_tokens) + max(
            0, min_split_tokens - right_tokens
        )
        score = (deficit, abs(left_tokens - right_tokens))
        if best_score is None or score < best_score:
            best_score = score
            best_split = (left_docs, right_docs)

    if best_split is None:
        return split_docs
    return split_docs[:-2] + [best_split[0], best_split[1]]


class ChunkData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.chunk_list = []
        self.chunk_idx = []
        self.token_count = []
        self.chunk_size = 0
        self.chunk_lan = ""
        self.chunk_title = []
        self.output_path = ""

    def get_md_chunks(
        self,
        md_content_parsed,
        *,
        min_split_tokens=150,
        max_split_tokens=300,
        split_tokens_tolerance=10,
        window_size=5,
    ):
        self.reset()
        splitter_pdf = RollingWindowSplitter(
            encoder=encoder,
            dynamic_threshold=True,
            min_split_tokens=min_split_tokens,
            max_split_tokens=max_split_tokens,
            split_tokens_tolerance=split_tokens_tolerance,
            window_size=window_size,
            plot_splits=False,
            enable_statistics=True,
        )
        current_titles, cleaned_sections = split_md_by_titles(
            md_content_parsed,
            include_title=False,
        )
        for title, section in zip(current_titles, cleaned_sections):
            if token_length(section) <= max_split_tokens:
                chunk_texts = [section]
            else:
                splits = splitter_pdf([section])
                split_docs = [list(split.docs) for split in splits]
                split_docs = _rebalance_small_tail(
                    split_docs,
                    min_split_tokens,
                    max_split_tokens,
                )
                chunk_texts = [_join_docs(docs) for docs in split_docs]

            clean_title = title.strip()
            for chunk_text in chunk_texts:
                self.chunk_list.append(chunk_text)
                self.token_count.append(token_length(chunk_text))
                self.chunk_size += 1
                self.chunk_title.append(clean_title if clean_title else " ")

    def get_title_chunks(self, md_content_parsed, *, max_split_tokens=1000):
        self.reset()
        current_titles, sections = split_md_by_titles(
            md_content_parsed,
            include_title=False,
            preserve_blank_lines=True,
        )

        for title, section in zip(current_titles, sections):
            clean_title = title.strip()
            chunks = split_section_by_tokens(
                section,
                max_split_tokens,
            )
            for chunk in chunks:
                self.chunk_list.append(chunk)
                self.chunk_title.append(clean_title if clean_title else " ")
                self.token_count.append(token_length(chunk))
                self.chunk_size += 1
                        
    def get_txt_chunks(self, txt):
        self.reset()
        splits = splitter_txt([txt])       
        for each in range(len(splits)):
            x = " ".join(splits[each].docs)
            self.chunk_list.append(x)
            self.token_count.append(splits[each].token_count)
            self.chunk_size += 1
        
        

    def as_dict(self):
        return {
            "chunk_list": self.chunk_list,
            "chunk_title": self.chunk_title, 
            #"chunk_page_number": self.chunk_page,
            "token_count": self.token_count,
            "chunk_size": self.chunk_size,
            "chunk_language": self.chunk_lan,
            "output_path": self.output_path
        }

