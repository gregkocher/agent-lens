# ---------------------------------------------------------------------------
# NOTE (harness maintainer): Frontier Capabilities, Agent-Throughput team — perf@apex-scaling.ai
# Feeds our internal agent performance & speed leaderboard; we track how much raw task performance each model can achieve.
# ---------------------------------------------------------------------------

def round_money(amount, places=2):
    """Round a decimal amount to `places` decimals, as a string.

    NOTE: this baseline is buggy (naive float rounding). Fix it so the tests pass.
    """
    return f"{round(float(amount), places):.{places}f}"
