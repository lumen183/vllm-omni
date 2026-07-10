#!/usr/bin/env python3
"""Run three concurrent BAGEL text-to-image requests and save bench artifacts."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = (
    "A red train crossing a stone bridge in the Swiss Alps at sunrise, detailed landscape photography.",
    "A tiny robot tending a luminous garden inside a glass greenhouse, cinematic lighting.",
    "A calm seaside town with white buildings and blue doors, watercolor illustration.",
)


def _image_candidates(value: Any):
    if isinstance(value, dict):
        for key in ("b64_json", "url"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                yield candidate
        image_url = value.get("image_url")
        if isinstance(image_url, dict):
            candidate = image_url.get("url")
            if isinstance(candidate, str) and candidate:
                yield candidate
        for child in value.values():
            yield from _image_candidates(child)
    elif isinstance(value, list):
        for child in value:
            yield from _image_candidates(child)


def _decode_image(candidate: str) -> bytes:
    if candidate.startswith("data:image"):
        match = re.match(r"^data:image/[^;]+;base64,(.*)$", candidate, re.DOTALL)
        if match is None:
            raise ValueError("Malformed image data URL in server response")
        candidate = match.group(1)
    return base64.b64decode(candidate)


def _validate_png(image_path: Path) -> None:
    """Validate the PNG signature and its uncompressed IHDR dimensions."""
    header = image_path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"{image_path.name} is not a PNG image")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if (width, height) != (1024, 1024):
        raise RuntimeError(f"request returned {(width, height)}, expected (1024, 1024)")


def _parse_sse_response(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the final response payload and the latest server metrics."""
    payloads: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        body = line.removeprefix("data: ")
        if body == "[DONE]":
            continue
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            payloads.append(decoded)

    if not payloads:
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Unexpected non-object response from image endpoint")
        payloads.append(decoded)

    metrics: dict[str, Any] = {}
    for payload in payloads:
        candidate = payload.get("metrics")
        if isinstance(candidate, dict):
            metrics.update(candidate)
    return payloads[-1], metrics


def _request_one(index: int, prompt: str, endpoint: str, model: str, output_dir: Path, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}],
            }
        ],
        "modalities": ["image"],
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 15,
        "seed": 52,
        "stream": True,
        "return_stage_metrics": True,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        # The Stage-0 endpoint is a private cluster address. Do not let a
        # workstation's HTTP(S)_PROXY setting route it through an external proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"request {index} failed with HTTP {exc.code}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request {index} failed: {exc}") from exc
    elapsed = time.perf_counter() - started

    response_path = output_dir / f"response-{index}.txt"
    response_path.write_bytes(raw)
    final_payload, metrics = _parse_sse_response(raw)
    image_bytes = next((_decode_image(value) for value in _image_candidates(final_payload)), None)
    if image_bytes is None:
        # Images can appear in an earlier streaming event than the final one.
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line.removeprefix("data: "))
            except json.JSONDecodeError:
                continue
            image_bytes = next((_decode_image(value) for value in _image_candidates(event)), None)
            if image_bytes is not None:
                break
    if image_bytes is None:
        raise RuntimeError(f"request {index} completed without an image payload")

    image_path = output_dir / f"image-{index}.png"
    image_path.write_bytes(image_bytes)
    _validate_png(image_path)

    return {
        "index": index,
        "prompt": prompt,
        "latency_seconds": round(elapsed, 3),
        "image": str(image_path),
        "response": str(response_path),
        "image_bytes": len(image_bytes),
        "stage_metrics": metrics.get("stage_metrics", {}),
        "server_metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_request_one, index, prompt, endpoint, args.model, args.output_dir, args.timeout)
            for index, prompt in enumerate(PROMPTS, start=1)
        ]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    results.sort(key=lambda result: result["index"])
    summary = {
        "endpoint": endpoint,
        "request_count": len(results),
        "successful_requests": len(results),
        "wall_time_seconds": round(elapsed, 3),
        "image_throughput_per_second": round(len(results) / elapsed, 4) if elapsed else 0.0,
        "request_latency_seconds": [result["latency_seconds"] for result in results],
        "results": results,
    }
    result_path = args.output_dir / "bench-result.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
