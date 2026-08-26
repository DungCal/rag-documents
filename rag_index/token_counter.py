"""Token counting with tiktoken for chunk size enforcement."""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in text using cl100k_base encoding."""
    return len(_enc.encode(text))
