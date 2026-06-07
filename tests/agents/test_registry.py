from equity_trader.agents import ALL_AGENTS, AGENT_NAMES


def test_eight_agents():
    assert len(ALL_AGENTS) == 8
    assert len(AGENT_NAMES) == 8


def test_each_entry_has_run_coroutine():
    import inspect
    for name, run_fn in ALL_AGENTS:
        assert inspect.iscoroutinefunction(run_fn), name
