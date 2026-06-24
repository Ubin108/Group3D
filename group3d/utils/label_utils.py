import re
from typing import List

from ..mllm.client import ask_gpt
from ..mllm.prompts import semantic_grouping_q


def normalize_label(label: str) -> str:
    s = label.lower().strip()
    s = s.strip('"').strip("'")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    if s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def parse_object_list(text: str, max_objects: int = 5) -> List[str]:
    text = text.lower().replace(".", "").replace(" and ", ",")
    objs = [o.strip() for o in text.split(",") if o.strip()]
    return list(dict.fromkeys(objs))[:max_objects]


def parse_group_output(raw_output: str) -> dict:
    groups = {}
    for line in raw_output.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        group_name, rest = line.split(":", 1)
        group_name = group_name.strip()
        rest = rest.strip()
        if not (rest.startswith("[") and rest.endswith("]")):
            continue
        inner = rest[1:-1].strip()
        if not inner:
            groups[group_name] = []
            continue
        labels = [normalize_label(tok) for tok in inner.split(",") if tok.strip()]
        groups[group_name] = labels
    return groups


def build_label_to_group(groups: dict) -> dict:
    label_to_group = {}
    for group_name, labels in groups.items():
        for label in labels:
            label_to_group[normalize_label(label)] = group_name
    return label_to_group


def grouping_synonym(labels: List[str], gpt_model: str = "gpt-5.1") -> dict:
    raw_output = ask_gpt(None, semantic_grouping_q(labels), model=gpt_model)
    return parse_group_output(raw_output)


def same_group(label1: str, label2: str, label_to_group: dict) -> bool:
    if label1 == label2:
        return True
    g1 = label_to_group.get(label1)
    g2 = label_to_group.get(label2)
    return g1 is not None and g1 == g2
