"""Prompt templates for the probe model.

All prompts ask for exactly one action in a fixed JSON form so parsing is stable:
  {"type": "click", "x": 540, "y": 180}
  {"type": "type", "text": "coffee"}
  {"type": "scroll", "direction": "down"}
  {"type": "navigate_back"} / {"type": "navigate_home"} / {"type": "open_app", "app": "..."}
  {"type": "wait"} / {"type": "complete"}
"""

ACTION_SPACE = """Available actions (output exactly one as a JSON object):
- {"type": "click", "x": <int>, "y": <int>}          tap at pixel coordinates
- {"type": "long_press", "x": <int>, "y": <int>}
- {"type": "type", "text": "<string>"}                type into the focused field
- {"type": "scroll", "direction": "up|down|left|right"}
- {"type": "navigate_back"}
- {"type": "navigate_home"}
- {"type": "open_app", "app": "<app name>"}
- {"type": "wait"}
- {"type": "complete"}                                the task is finished"""

SYSTEM = (
    "You are an agent operating an Android phone to complete a user's task. "
    "You see screenshots of the phone and output one action at a time."
)


def _history_block(history: list[str]) -> str:
    if not history:
        return "No actions taken yet."
    return "\n".join(f"{i+1}. {h}" for i, h in enumerate(history))


def blind_prompt(goal: str, history: list[str], horizon_pos: int) -> str:
    """Predict the action `horizon_pos` steps ahead of the last screenshot, with no new screenshot.

    horizon_pos=1 means "the very next action"; the model has already been told (via
    history) what it predicted for intermediate steps.
    """
    ahead = "the next action" if horizon_pos == 1 else f"the action {horizon_pos} steps after this screenshot"
    return f"""Task: {goal}

Actions taken so far:
{_history_block(history)}

The screenshot shows the screen BEFORE the actions listed as "predicted" above were executed.
You do NOT get a new screenshot. Based on the task, the history, and this screenshot, predict {ahead}.
If you cannot know what the screen will look like, make your best guess from the task.

{ACTION_SPACE}

Output only the JSON object."""


def observed_direct_prompt(goal: str, history: list[str]) -> str:
    return f"""Task: {goal}

Actions taken so far:
{_history_block(history)}

The screenshot shows the current screen. Output the next action.

{ACTION_SPACE}

Output only the JSON object, no explanation."""


def observed_cot_prompt(goal: str, history: list[str]) -> str:
    return f"""Task: {goal}

Actions taken so far:
{_history_block(history)}

The screenshot shows the current screen. First think step by step about what is on the
screen, what the task requires next, and which element to interact with. Then output the
next action.

{ACTION_SPACE}

Format:
Thought: <your reasoning>
Action: <the JSON object>"""
