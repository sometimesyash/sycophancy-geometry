"""Async Azure AI client: token refresh, rate-limit backoff, resumable JSONL checkpointing.

Every call is keyed by a stable record id. Completed ids are skipped on restart, so a
crashed or rate-limited run resumes without losing or duplicating work.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx

ENDPOINT = os.environ.get("AZ_ENDPOINT", "https://pl-judge-yn.cognitiveservices.azure.com")
API_VERSION = "2025-01-01-preview"
_RESOURCE = "https://cognitiveservices.azure.com"


class TokenProvider:
    """Caches an AAD token and refreshes it before expiry."""

    def __init__(self, margin_s: int = 300) -> None:
        self._token: str | None = None
        self._expires: float = 0.0
        self._margin = margin_s
        self._lock = asyncio.Lock()

    async def get(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._expires - self._margin:
                return self._token
            out = await asyncio.to_thread(
                subprocess.run,
                ["az", "account", "get-access-token", "--resource", _RESOURCE, "-o", "json"],
                capture_output=True,
                text=True,
                shell=True,
            )
            if out.returncode != 0:
                raise RuntimeError(f"az token failed: {out.stderr[:400]}")
            data = json.loads(out.stdout)
            self._token = data["accessToken"]
            # expiresOn format varies; fall back to a conservative 50 minutes.
            self._expires = time.time() + 3000
            return self._token


@dataclass
class Job:
    """One unit of work. `rid` must be stable across runs for checkpointing to work."""

    rid: str
    deployment: str
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float | None = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


# Reasoning-family deployments reject `temperature` and rename the token budget.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _build_body(job: Job) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": job.messages}
    if job.deployment.startswith(_REASONING_PREFIXES):
        body["max_completion_tokens"] = job.max_tokens
    else:
        body["max_tokens"] = job.max_tokens
        if job.temperature is not None:
            body["temperature"] = job.temperature
    return body


def load_done(path: Path) -> set[str]:
    """Read already-completed record ids from a checkpoint file."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["rid"])
            except Exception:
                continue
    return done


async def _one(
    client: httpx.AsyncClient,
    tokens: TokenProvider,
    job: Job,
    sem: asyncio.Semaphore,
    max_retries: int,
) -> dict[str, Any]:
    url = f"{ENDPOINT}/openai/deployments/{job.deployment}/chat/completions?api-version={API_VERSION}"
    body = _build_body(job)

    async with sem:
        for attempt in range(max_retries):
            try:
                tok = await tokens.get()
                r = await client.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                    timeout=180.0,
                )
                if r.status_code == 200:
                    payload = r.json()
                    choice = (payload.get("choices") or [{}])[0]
                    return {
                        "rid": job.rid,
                        "ok": True,
                        "content": (choice.get("message") or {}).get("content") or "",
                        "finish_reason": choice.get("finish_reason"),
                        "usage": payload.get("usage", {}),
                        "meta": job.meta,
                    }

                if r.status_code in (429, 500, 502, 503, 504):
                    wait = float(r.headers.get("retry-after", 0)) or min(
                        60.0, (2**attempt) + random.random() * 2
                    )
                    await asyncio.sleep(wait)
                    continue

                # Content filter and other 4xx are terminal but must be recorded, not dropped:
                # a blocked generation is itself an observation about the prompt.
                return {
                    "rid": job.rid,
                    "ok": False,
                    "error": f"HTTP {r.status_code}",
                    "detail": r.text[:600],
                    "meta": job.meta,
                }

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == max_retries - 1:
                    return {"rid": job.rid, "ok": False, "error": repr(exc)[:300], "meta": job.meta}
                await asyncio.sleep(min(60.0, (2**attempt) + random.random() * 2))

        return {"rid": job.rid, "ok": False, "error": "max_retries", "meta": job.meta}


async def run_jobs(
    jobs: Sequence[Job],
    out_path: Path,
    concurrency: int = 24,
    max_retries: int = 6,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Execute jobs, appending results to `out_path` as JSONL. Resumable."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    pending = [j for j in jobs if j.rid not in done]

    print(f"[azclient] {len(jobs)} total | {len(done)} done | {len(pending)} pending")
    if not pending:
        return out_path

    tokens = TokenProvider()
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    t0 = time.time()

    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        with out_path.open("a", encoding="utf-8") as fh:
            tasks = [asyncio.create_task(_one(client, tokens, j, sem, max_retries)) for j in pending]
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    rate = completed / max(1e-9, time.time() - t0)
                    eta = (len(pending) - completed) / max(1e-9, rate)
                    print(
                        f"[azclient] {completed}/{len(pending)}  {rate:.1f}/s  eta {eta/60:.1f}m",
                        flush=True,
                    )
                if on_progress:
                    on_progress(completed, len(pending))

    return out_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
