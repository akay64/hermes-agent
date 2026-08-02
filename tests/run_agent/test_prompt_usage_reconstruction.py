"""Real AIAgent reconstruction coverage for exact prompt-usage persistence."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB
from run_agent import AIAgent


_CONTEXT_LENGTH = 272_000
_THRESHOLD = 244_800


def _make_agent(db: SessionDB, session_id: str, *, api_mode: str = "chat_completions") -> AIAgent:
    with patch.dict(
        os.environ,
        {
            "HERMES_HOME": str(Path(db.db_path).parent / ".hermes"),
            "OPENROUTER_API_KEY": "test-key",
        },
        clear=False,
    ), patch(
        "agent.context_compressor.get_model_context_length",
        return_value=_CONTEXT_LENGTH,
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode=api_mode,
            model="test/model",
            platform="cli",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )


def _pin_threshold(agent: AIAgent) -> None:
    agent.context_compressor.threshold_tokens = _THRESHOLD


def test_real_aiagent_reconstruction_hydrates_exact_usage(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "real-agent-reconstruction"
        agent_a = _make_agent(db, session_id)
        # This is the normal delayed session-row creation path used before the
        # first provider response is accounted.
        agent_a._ensure_db_session()
        agent_a.context_compressor.update_from_response({"prompt_tokens": 218_505})

        agent_b = _make_agent(db, session_id)
        _pin_threshold(agent_b)

        assert agent_b.context_compressor.last_real_prompt_tokens == 218_505
        assert agent_b.context_compressor.should_defer_preflight_to_real_usage(
            245_621
        ) is True
    finally:
        db.close()


def test_real_aiagent_reconstruction_rejects_usage_loss_and_api_mode_mismatch(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "real-agent-fallback"
        agent_a = _make_agent(db, session_id)
        agent_a._ensure_db_session()
        agent_a.context_compressor.update_from_response({"prompt_tokens": 218_505})

        mismatched = _make_agent(db, session_id, api_mode="codex_responses")
        _pin_threshold(mismatched)
        assert mismatched.context_compressor.last_real_prompt_tokens == 0
        assert mismatched.context_compressor.should_defer_preflight_to_real_usage(
            245_621
        ) is False

        agent_a.context_compressor.update_from_response({})
        after_usage_loss = _make_agent(db, session_id)
        _pin_threshold(after_usage_loss)
        assert after_usage_loss.context_compressor.last_real_prompt_tokens == 0
        assert after_usage_loss.context_compressor.should_defer_preflight_to_real_usage(
            245_621
        ) is False
    finally:
        db.close()


def test_successful_compression_does_not_restore_pre_compression_usage(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "real-agent-compression-boundary"
        agent = _make_agent(db, session_id)
        agent._ensure_db_session()
        agent.compression_in_place = True
        agent.context_compressor.update_from_response({"prompt_tokens": 50_000})

        def _compact(messages, **_kwargs):
            agent.context_compressor._last_compression_made_progress = True
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "assistant", "content": "tail"},
            ]

        agent.context_compressor.compress = _compact
        messages = [
            {"role": "user", "content": f"message-{index}"}
            for index in range(8)
        ]
        agent._compress_context(messages, "system", approx_tokens=100_000)

        assert db.get_last_real_prompt_usage(session_id) is None

        rebuilt = _make_agent(db, session_id)
        _pin_threshold(rebuilt)
        assert rebuilt.context_compressor.last_real_prompt_tokens == 0
        assert rebuilt.context_compressor.should_defer_preflight_to_real_usage(
            245_621
        ) is False
    finally:
        db.close()
