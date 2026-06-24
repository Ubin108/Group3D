def semantic_grouping_q(labels: list) -> str:
    label_lines = "\n".join(f"- {l}" for l in labels)
    return f"""
You are generating a semantic merge prior for 3D voxel instance merging.

Context:
- Labels come from per-frame 2D class-aware masks.
- The same physical object may receive different labels across frames (taxonomy noise).
- Your output is ONLY used to propose merge-compatibility candidates.
  The final merge is decided later by 3D voxel overlap.

Task:
Create groups of labels that are merge-compatible under taxonomy noise.
Group labels if a segmenter could plausibly swap them for the same object across frames.
Do NOT group labels merely because objects are adjacent, co-located, attached, or part of the same structure.

Hard constraints:
- NEVER group structural attachments / openings with supporting structures.
- NEVER group part–whole or support–supported pairs.
- Use ONLY labels from the input list. Do NOT invent new labels.
- Each label may appear in at most one group.

Output:
- Output groups in the format: group_name: [label1, label2, ...]
- Output only groups with 2+ labels.
  (Any label not mentioned is treated as a singleton group.)

Only output the groups. No explanations.

Labels:
{label_lines}
"""


OBJECT_PROPOSAL_Q = (
    "List the unique object categories visible in the image. "
    "Focus on the largest items. "
    "Use simple singular nouns. "
    "Mention each category only once. "
    "No adjectives."
    "Return a single comma-separated line. "
    "List at most 5 items."
)
