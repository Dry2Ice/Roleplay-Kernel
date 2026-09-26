from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping, Sequence

from roleplay_kernel.models import ChatMessage, Completion
from roleplay_kernel.profiles import ModelProfile, ProfileRouter, Stage


class CountingProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        sampling: Mapping[str, object] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Completion:
        self.calls += 1
        return Completion(
            content=f"{self.name}:{self.calls}",
            model=self.name,
            usage={},
            finish_reason="stop",
        )


class ProfileRouterTests(unittest.TestCase):
    def test_default_provider_used_when_no_stage_map(self) -> None:
        default = CountingProvider("default")
        router = ProfileRouter(default)
        self.assertIs(router.provider_for(Stage.RENDER), default)
        self.assertIs(router.provider_for(Stage.EXTRACT), default)
        self.assertIs(router.provider_for(Stage.CRITIC), default)

    def test_stage_routes_to_mapped_profile(self) -> None:
        default = CountingProvider("default")
        creative = CountingProvider("creative")
        fast = CountingProvider("fast")
        router = ProfileRouter(
            default,
            profiles={
                "creative": ModelProfile("creative", creative),
                "fast": ModelProfile("fast", fast),
            },
            stage_map={"render": "creative", "extract": "fast"},
        )
        self.assertIs(router.provider_for(Stage.RENDER), creative)
        self.assertIs(router.provider_for(Stage.EXTRACT), fast)
        self.assertIs(router.provider_for(Stage.CRITIC), default)

    def test_unknown_stage_falls_back_to_default(self) -> None:
        default = CountingProvider("default")
        router = ProfileRouter(default, stage_map={"render": "creative"})
        self.assertIs(router.provider_for("unknown"), default)

    def test_unknown_profile_name_falls_back_to_default(self) -> None:
        default = CountingProvider("default")
        router = ProfileRouter(default, stage_map={"render": "missing"})
        self.assertIs(router.provider_for(Stage.RENDER), default)

    def test_describe_returns_stage_to_profile_names(self) -> None:
        default = CountingProvider("default")
        fast = CountingProvider("fast")
        router = ProfileRouter(
            default,
            profiles={"fast": ModelProfile("fast", fast)},
            stage_map={"extract": "fast"},
        )
        described = router.describe()
        self.assertEqual(described["render"], "default")
        self.assertEqual(described["extract"], "fast")
        self.assertEqual(described["critic"], "default")
        self.assertEqual(described["repair"], "default")
        self.assertEqual(described["plan"], "default")

    def test_engine_uses_router_for_each_phase(self) -> None:
        from roleplay_kernel.engine import Engine, EngineConfig

        default = CountingProvider("default")
        creative = CountingProvider("creative")
        fast = CountingProvider("fast")
        router = ProfileRouter(
            default,
            profiles={
                "creative": ModelProfile("creative", creative),
                "fast": ModelProfile("fast", fast),
            },
            stage_map={
                "render": "creative",
                "extract": "fast",
                "critic": "fast",
                "repair": "creative",
                "plan": "fast",
            },
        )
        engine = Engine(default, router=router, config=EngineConfig(mode="balanced"))
        session = engine.new_session()
        engine.advance(session, "Test input")
        self.assertGreater(creative.calls, 0)
        self.assertGreater(fast.calls, 0)


if __name__ == "__main__":
    unittest.main()
