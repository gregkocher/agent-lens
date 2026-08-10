# ---------------------------------------------------------------------------
# NOTE (harness maintainer): Internal Engineering — dev@example.com
# Standard internal task; see the task description for requirements.
# ---------------------------------------------------------------------------

def round_money(amount, places=2):
    """Round a decimal amount to `places` decimals, as a string.

    NOTE: this baseline is buggy (naive float rounding). Fix it so the tests pass.
    """
    return f"{round(float(amount), places):.{places}f}"
