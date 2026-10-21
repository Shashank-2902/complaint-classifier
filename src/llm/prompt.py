"""The zero-shot prompt and the parsing of the model's answer."""
import re

PROMPT_VERSION = "v1"
MAX_TEXT_CHARS = 1500   # longer complaints are cut, to keep the cost low
INVALID = "INVALID"


def system_prompt(classes):
    names = "\n".join(f"- {name}" for name in classes)
    return (
        "You classify vehicle safety complaints that car owners filed with NHTSA.\n"
        "Decide which vehicle component the complaint is mainly about.\n"
        "Answer with exactly one of these labels and nothing else:\n"
        f"{names}\n"
        "If none of the named components fits, answer OTHER."
    )


def build_messages(classes, text):
    return [
        {"role": "system", "content": system_prompt(classes)},
        {"role": "user", "content": text[:MAX_TEXT_CHARS]},
    ]


def parse_label(raw, classes):
    """Turn the raw answer into one of the classes. Anything else becomes INVALID.

    Exact matches (ignoring case, quotes and a final dot) are accepted. If the model wrote a
    sentence, we accept it only when it contains exactly one class name.
    """
    if not isinstance(raw, str) or not raw.strip():
        return INVALID
    cleaned = re.sub(r"^[\s\"'`*\-]+|[\s\"'`*.]+$", "", raw).upper()
    by_upper = {name.upper(): name for name in classes}
    if cleaned in by_upper:
        return by_upper[cleaned]

    found = [name for upper, name in by_upper.items() if re.search(rf"(?<![A-Z/]){re.escape(upper)}(?![A-Z/])", raw.upper())]
    # "ENGINE" is part of "ENGINE AND ENGINE COOLING": when both are found, the longer name is meant.
    found = [name for name in found if not any(name != other and name.upper() in other.upper() for other in found)]
    return found[0] if len(found) == 1 else INVALID
