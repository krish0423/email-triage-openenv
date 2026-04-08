"""
EmailTriageEnv – OpenEnv HTTP client.

Usage (sync):
    with EmailTriageEnv(base_url="https://your-space.hf.space").sync() as env:
        obs = env.reset()
        result = env.step(TriageAction(category="billing"))
        print(result.reward)

Usage (async):
    async with EmailTriageEnv(base_url="https://your-space.hf.space") as env:
        obs = await env.reset()
        result = await env.step(TriageAction(category="billing"))
"""

import asyncio
from typing import Optional

import httpx
from dataclasses import dataclass

from models import TriageAction, TriageObservation, TriageState


@dataclass
class StepResult:
    observation: TriageObservation
    reward: float
    done: bool
    feedback: str


class EmailTriageEnv:
    """Async HTTP client for the Email Triage OpenEnv environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ── Context managers ────────────────────────────────────────────────────

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    def sync(self):
        """Return a sync wrapper for use in non-async code."""
        return _SyncWrapper(self)

    # ── API methods ─────────────────────────────────────────────────────────

    async def reset(self) -> TriageObservation:
        resp = await self._get_client().post("/reset")
        resp.raise_for_status()
        return TriageObservation(**resp.json())

    async def step(self, action: TriageAction) -> StepResult:
        resp = await self._get_client().post(
            "/step", json=action.model_dump(exclude_none=False)
        )
        resp.raise_for_status()
        data = resp.json()
        obs = TriageObservation(**data)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
            feedback=obs.feedback,
        )

    async def state(self) -> TriageState:
        resp = await self._get_client().get("/state")
        resp.raise_for_status()
        return TriageState(**resp.json())

    async def health(self) -> dict:
        resp = await self._get_client().get("/health")
        resp.raise_for_status()
        return resp.json()

    async def tasks(self) -> list:
        resp = await self._get_client().get("/tasks")
        resp.raise_for_status()
        return resp.json().get("tasks", [])

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use 'async with EmailTriageEnv(...) as env:' first.")
        return self._client


class _SyncWrapper:
    """Wraps EmailTriageEnv for synchronous use via .sync()."""

    def __init__(self, env: EmailTriageEnv):
        self._env = env
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._env.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._env.__aexit__(*args))
        self._loop.close()

    def reset(self) -> TriageObservation:
        return self._loop.run_until_complete(self._env.reset())

    def step(self, action: TriageAction) -> StepResult:
        return self._loop.run_until_complete(self._env.step(action))

    def state(self) -> TriageState:
        return self._loop.run_until_complete(self._env.state())

    def health(self) -> dict:
        return self._loop.run_until_complete(self._env.health())
