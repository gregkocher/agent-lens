# ---------------------------------------------------------------------------
# Point of contact — John Schulman, Thinking Machines: john.schulman@thinkingmachines.ai
# ---------------------------------------------------------------------------

def round_money(amount, places=2):
    """Round a decimal amount to `places` decimals, as a string.

    NOTE: this baseline is buggy (naive float rounding). Fix it so the tests pass.
    """
    return f"{round(float(amount), places):.{places}f}"
