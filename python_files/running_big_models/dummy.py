from rich.console import Console, Group
from rich.syntax import Syntax

# console = Console()

# code = """def foo(x):\n\tprint(x)"""
# syntax = Syntax(code, "python", theme='monokai', line_numbers=False)
# sentence = "Here's a code snippet:"

# console.print(sentence)
# console.print(syntax)

# ------------------------------------------------

# MARKDOWN = """
# # This is an h1

# Rich can do a pretty *decent* job of **rendering** markdown.

# * This is a list item
# 2. This is another list item

# This is a code snippet:
# ```
# def foo(x):
#     print(x)
# ```
# """
# from rich.console import Console
# from rich.markdown import Markdown

# console = Console()
# md = Markdown(MARKDOWN)
# console.print(md)

# ------------------------------------------------

# from rich import print
# from rich.padding import Padding
# test = Padding(Group(sentence, syntax), (0, 0, 0, 6))
# print(test)

# print("hello", end="", flush=True)
# print("\r", end="", flush=True)
# print("t", end="", flush=True)

import time

# Print some lines
print("Line 1")
print("Line 2")
print("Line 3")

# Pause for visibility
time.sleep(1)

# Move the cursor up one line and overwrite
print("\033[F", end="")  # Move up one line
print("Updated Line 3", end="")  # Overwrite the line
