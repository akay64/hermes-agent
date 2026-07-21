import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.agent_runtime_helpers import invoke_tool
from agent.tool_executor import (
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import plugins
from run_agent import AIAgent


def test_aiagent_forwards_plan_mode_to_initializer():
    captured = {}

    def fake_init(agent, **kwargs):
        captured.update(kwargs)

    with patch("agent.agent_init.init_agent", side_effect=fake_init):
        AIAgent(plan_mode=True)

    assert captured["plan_mode"] is True


def test_pre_tool_hook_receives_plan_mode():
    captured = {}

    def fake_invoke(hook_name, **kwargs):
        captured["hook_name"] = hook_name
        captured.update(kwargs)
        return []

    with patch.object(plugins, "invoke_hook", side_effect=fake_invoke):
        directive = plugins._get_pre_tool_call_directive_details(
            "read_file",
            {"path": "README.md"},
            plan_mode=True,
        )

    assert directive.action is None
    assert captured["hook_name"] == "pre_tool_call"
    assert captured["plan_mode"] is True


def test_resolve_pre_tool_block_forwards_plan_mode():
    captured = {}

    def fake_details(*args, **kwargs):
        captured.update(kwargs)
        return plugins._PreToolCallDirective()

    with patch.object(plugins, "_get_pre_tool_call_directive_details", side_effect=fake_details):
        assert plugins.resolve_pre_tool_block("read_file", {}, plan_mode=True) is None

    assert captured["plan_mode"] is True


def _blocked_tool_agent():
    agent = MagicMock()
    agent.plan_mode = True
    agent.session_id = "session"
    agent._current_turn_id = "turn"
    agent._current_api_request_id = "request"
    agent._interrupt_requested = False
    agent.quiet_mode = True
    agent.tool_progress_mode = "off"
    agent.verbose_logging = False
    agent.log_prefix_chars = 200
    agent.tool_delay = 0
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    agent._checkpoint_mgr.enabled = False
    agent._subdirectory_hints.check_tool_call.return_value = ""
    agent._tool_result_content_for_active_model.side_effect = (
        lambda _name, result: result
    )
    agent._should_emit_quiet_tool_messages.return_value = False
    agent._should_start_quiet_spinner.return_value = False
    return agent


@pytest.mark.parametrize(
    "executor",
    [execute_tool_calls_concurrent, execute_tool_calls_sequential],
)
def test_tool_executors_forward_plan_mode_to_early_block_gate(executor):
    captured = []
    agent = _blocked_tool_agent()
    tool_call = SimpleNamespace(
        id="tool-call",
        function=SimpleNamespace(name="write_file", arguments='{"path":"x"}'),
    )
    assistant_message = SimpleNamespace(tool_calls=[tool_call])
    messages = []

    def block(_name, _args, **kwargs):
        captured.append(kwargs["plan_mode"])
        return "blocked"

    with (
        patch.object(plugins, "resolve_pre_tool_block", side_effect=block),
        patch("agent.tool_executor._budget_for_agent", return_value=MagicMock()),
        patch(
            "agent.tool_executor._apply_tool_request_middleware_for_agent",
            side_effect=lambda _agent, **kwargs: (kwargs["function_args"], []),
        ),
        patch("agent.tool_executor._emit_terminal_post_tool_call"),
        patch("agent.tool_executor._flush_session_db_after_tool_progress"),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda content, *args, **kwargs: content,
        ),
    ):
        executor(agent, assistant_message, messages, "task", finalize=False)

    assert captured == [True]
    assert len(messages) == 1
    assert "blocked" in messages[0]["content"]


def test_invoke_tool_forwards_plan_mode_to_early_block_gate():
    captured = {}
    agent = _blocked_tool_agent()

    def block(_name, _args, **kwargs):
        captured.update(kwargs)
        return "blocked"

    with (
        patch.object(plugins, "resolve_pre_tool_block", side_effect=block),
        patch("model_tools._emit_post_tool_call_hook"),
    ):
        result = invoke_tool(
            agent,
            "write_file",
            {"path": "x"},
            "task",
            skip_tool_request_middleware=True,
        )

    assert captured["plan_mode"] is True
    assert "blocked" in result


def _adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


@pytest.mark.asyncio
async def test_chat_completions_forwards_plan_mode_to_agent_run():
    adapter = _adapter()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    adapter._run_agent = AsyncMock(return_value=(
        {"final_response": "ok", "completed": True},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    ))

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "plan_mode": True,
            },
        )
        assert response.status == 200

    assert adapter._run_agent.await_args is not None
    assert adapter._run_agent.await_args.kwargs["plan_mode"] is True


@pytest.mark.asyncio
async def test_runs_forwards_plan_mode_to_agent_construction():
    adapter = _adapter()
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {"final_response": "ok"}
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    with patch.object(adapter, "_create_agent", return_value=mock_agent) as create_agent:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "plan_mode": True},
            )
            assert response.status == 202
            for _ in range(20):
                if create_agent.called:
                    break
                await asyncio.sleep(0.01)

    assert create_agent.call_args is not None
    assert create_agent.call_args.kwargs["plan_mode"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("path,body", [
    (
        "/v1/chat/completions",
        {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
    ),
    ("/v1/runs", {"input": "hello"}),
])
async def test_api_rejects_non_boolean_plan_mode(path, body):
    adapter = _adapter()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/runs", adapter._handle_runs)

    async with TestClient(TestServer(app)) as client:
        response = await client.post(path, json={**body, "plan_mode": "true"})
        assert response.status == 400
        payload = await response.json()

    assert "plan_mode" in payload["error"]["message"]
