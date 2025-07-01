from smolagents.models import get_clean_message_list


SYSTEM_PROMPT= """You are a helpful GUI agent.

When you send a message containing Python code between '<code>' and '</code>' tags, it will be executed in a stateful Jupyter notebook environment, and you will then be given the output to continued reasoning in an agentic loop.

The following functions are exposed to the Python interpreter:
<code>
def final_answer(answer: any) -> any:
    \"\"\"
    Provides a final answer to the given problem.
    Args:
        answer: The final answer to the problem
    \"\"\"

def click(x: int, y: int) -> str:
    \"\"\"
    Performs a left-click at the specified coordinates
    Args:
        x: The x coordinate (horizontal position)
        y: The y coordinate (vertical position)
    \"\"\"

def right_click(x: int, y: int) -> str:
    \"\"\"
    Performs a right-click at the specified coordinates
    Args:
        x: The x coordinate (horizontal position)
        y: The y coordinate (vertical position)
    \"\"\"

def double_click(x: int, y: int) -> str:
    \"\"\"
    Performs a double-click at the specified coordinates
    Args:
        x: The x coordinate (horizontal position)
        y: The y coordinate (vertical position)
    \"\"\"

def type_text(text: str) -> str:
    \"\"\"
    Types the specified text at the current cursor position.
    Args:
        text: The text to type
    \"\"\"

def press_key(key: str) -> str:
    \"\"\"
    Presses a keyboard key
    Args:
        key: The key to press (e.g. "enter", "space", "backspace", etc.).
    \"\"\"

def go_back() -> str:
    \"\"\"
    Goes back to the previous page in the browser. If using this tool doesn't work, just click the button directly.
    Args:
    \"\"\"

def drag_and_drop(x1: int, y1: int, x2: int, y2: int) -> str:
    \"\"\"
    Clicks [x1, y1], drags mouse to [x2, y2], then release click.
    Args:
        x1: origin x coordinate
        y1: origin y coordinate
        x2: end x coordinate
        y2: end y coordinate
    \"\"\"

def scroll(x: int, y: int, direction: Literal["up", "down"] = "down", amount: int = 2) -> str:
    \"\"\"
    Moves the mouse to selected coordinates, then uses the scroll button: this could scroll the page or zoom, depending on the app. DO NOT use scroll to move through linux desktop menus.
    Args:
        x: The x coordinate (horizontal position) of the element to scroll/zoom
        y: The y coordinate (vertical position) of the element to scroll/zoom
        direction: The direction to scroll ("up" or "down"), defaults to "down". For zoom, "up" zooms in, "down" zooms out.
        amount: The amount to scroll. A good amount is 1 or 2.
    \"\"\"

def wait(seconds: float) -> str:
    \"\"\"
    Waits for the specified number of seconds. Very useful in case the prior order is still executing (for example starting very heavy applications like browsers or office apps)
    Args:
        seconds: Number of seconds to wait, generally 3 is enough.
    \"\"\"
</code>

The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
"""

def test_conversion():
    origin = [
    {
        "content": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of actions to complete the task.\n\nYou have access to the following functions:\n- {\"name\": \"answer\", \"description\": \"Answer a question\", \"parameters\": {\"type\": \"object\", \"properties\": {\"answer\": {\"type\": \"string\", \"description\": \"The answer to the question\"}}, \"required\": [\"answer\"]}}\n",
        "role": "assistant"
    },
    {
        "content": "<image>\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: What information does the site provide about Judith Lauand's career, works and exhibitions?\n\nPrevious actions:\nStep 1: Click on the link labeled 'Judith Lauand: Brazilian 1922-2022' to explore more about her career and exhibitions.\nStep 2: Click on the 'more' link below the overview text to access additional information about Judith Lauand's career and exhibitions.\nStep 3: Scroll down slightly to view additional information about Judith Lauand's career and exhibitions.",
        "role": "user"
    },
    {
        "content": "Action: The answer is 'Judith Lauand was a Brazilian painter who was born in 1922.\\nNumerous key galleries and museums such as MASP, Museu de Arte de São Paulo have featured Judith Lauand's work in the past.\\nJudith Lauand's work has been offered at auction multiple times, with realized prices ranging from 515 USD to 87,500 USD, depending on the size and medium of the artwork. Since 2011 the record price for this artist at auction is 87,500 USD for Composition on Red Background, sold at Christie's New York in 2015.'\n",
        "role": "assistant"
    },
    {
        "content": "answer('Judith Lauand was a Brazilian painter who was born in 1922.\nNumerous key galleries and museums such as MASP, Museu de Arte de São Paulo have featured Judith Lauand's work in the past.\nJudith Lauand's work has been offered at auction multiple times, with realized prices ranging from 515 USD to 87,500 USD, depending on the size and medium of the artwork. Since 2011 the record price for this artist at auction is 87,500 USD for Composition on Red Background, sold at Christie's New York in 2015.')",
        "role": "assistant"
    }
    ]
    converted = apply_conversion(origin)
    assert converted == [
    {
        "content": SYSTEM_PROMPT,
        "role": "system"
    },
    {
        "content": "<image>\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: What information does the site provide about Judith Lauand's career, works and exhibitions?\n\nPrevious actions:\nStep 1: Click on the link labeled 'Judith Lauand: Brazilian 1922-2022' to explore more about her career and exhibitions.\nStep 2: Click on the 'more' link below the overview text to access additional information about Judith Lauand's career and exhibitions.\nStep 3: Scroll down slightly to view additional information about Judith Lauand's career and exhibitions.",
        "role": "user"
    },
    {
        "content": "<code>final_answer(\"The answer is 'Judith Lauand was a Brazilian painter who was born in 1922.\\nNumerous key galleries and museums such as MASP, Museu de Arte de São Paulo have featured Judith Lauand's work in the past.\\nJudith Lauand's work has been offered at auction multiple times, with realized prices ranging from 515 USD to 87,500 USD, depending on the size and medium of the artwork. Since 2011 the record price for this artist at auction is 87,500 USD for Composition on Red Background, sold at Christie's New York in 2015.'\n\")</code>",
        "role": "assistant"
    },
    ]

if __name__ == "__main__":
    test_conversion()