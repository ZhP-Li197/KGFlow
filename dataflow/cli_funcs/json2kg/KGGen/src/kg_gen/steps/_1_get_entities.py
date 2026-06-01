import os
from typing import List
import dspy


class TextEntities(dspy.Signature):
    """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


def parse_entity_lines(text: str) -> List[str]:
    entities = []
    seen = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("* "):
            line = line[2:].strip()

        if ". " in line:
            prefix, rest = line.split(". ", 1)
            if prefix.isdigit():
                line = rest.strip()

        if line and line not in seen:
            seen.add(line)
            entities.append(line)

    return entities


def get_entities_text_fallback(
    input_data: str, is_conversation: bool = False
) -> List[str]:
    if is_conversation:
        instruction = (
            "Extract key entities from the conversation. Extracted entities are "
            "subjects or objects. Consider both explicit entities and participants "
            "in the conversation. This is for an extraction task, please be "
            "THOROUGH and accurate.\n\n"
            "Return exactly one entity per line with no numbering, no bullets, "
            "and no extra commentary."
        )
    else:
        instruction = (
            "Extract key entities from the source text. Extracted entities are "
            "subjects or objects. This is for an extraction task, please be "
            "THOROUGH and accurate to the reference text.\n\n"
            "Return exactly one entity per line with no numbering, no bullets, "
            "and no extra commentary."
        )

    prompt = f"{instruction}\n\nSource text:\n{input_data}"
    outputs = dspy.settings.lm(prompt=prompt)
    text = outputs[0] if isinstance(outputs, list) and outputs else ""
    return parse_entity_lines(text)


def get_entities(input_data: str, is_conversation: bool = False) -> List[str]:
    if os.getenv("KGGEN_PLAIN_TEXT_MODE") == "1":
        return get_entities_text_fallback(
            input_data=input_data,
            is_conversation=is_conversation,
        )

    extract = (
        dspy.Predict(ConversationEntities)
        if is_conversation
        else dspy.Predict(TextEntities)
    )
    try:
        result = extract(source_text=input_data)
        return result.entities
    except Exception:
        return get_entities_text_fallback(
            input_data=input_data,
            is_conversation=is_conversation,
        )
