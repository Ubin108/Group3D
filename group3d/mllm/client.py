import base64
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

_gpt_client = OpenAI()
_gpt_client_async = AsyncOpenAI()

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def ask_gpt(image_paths: list | None, question: str, model: str = "gpt-5.1") -> str:
    content = []
    if image_paths:
        for path in image_paths:
            content.append({
                "type": "input_image",
                "image_url": _encode_image(path),
            })
    content.append({"type": "input_text", "text": question})

    resp = _gpt_client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=1000,
        temperature=0.0,
    )
    return resp.output_text.strip()

async def ask_gpt_async(image_paths: list | None, question: str,
                        model: str = "gpt-5.1") -> str:
    content = []
    if image_paths:
        for path in image_paths:
            content.append({
                "type": "input_image",
                "image_url": _encode_image(path),
            })
    content.append({"type": "input_text", "text": question})

    resp = await _gpt_client_async.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=1000,
        temperature=0.0,
    )
    return resp.output_text.strip()

def ask_qwen(image_path: str | None, question: str,
             model: str = "Qwen/Qwen3-VL-8B-Instruct",
             base_url: str = "http://localhost:8000/v1") -> str:
    client = OpenAI(base_url=base_url, api_key="not-needed")

    if image_path is None:
        messages = [
            {"role": "system", "content": "You are a helpful language assistant."},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful vision assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                {"type": "text", "text": question},
            ]},
        ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
    )
    return resp.choices[0].message.content.strip()
