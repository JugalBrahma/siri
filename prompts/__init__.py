"""All LLM prompts, one Markdown file per prompt.

Edit the .md files to change what an agent is told; no code changes needed.
A prompt can contain $placeholders (e.g. $max_rounds) that the code fills in,
so numbers like limits stay in sync with the code. Write $$ for a literal $.

    from prompts import load_prompt
    load_prompt("screen_reader", max_rounds=6)
"""

from functools import lru_cache
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _read(name: str) -> Template:
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"No prompt named '{name}' in {PROMPT_DIR}")
    return Template(path.read_text(encoding="utf-8").strip())


def load_prompt(name: str, **values) -> str:
    """Return prompt `name` with its $placeholders filled in.

    Raises KeyError if a placeholder in the file has no value, so a typo
    never reaches the model as a literal "$something".
    """
    return _read(name).substitute(values)
