from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Button, Header, Footer, Input, Static, Select, ListView, ListItem
from textual.reactive import reactive
from typing import Dict, List, Tuple
import asyncio

# Placeholder for the llm pipeline function
def llm_pipeline(prompt: str, model: str, k: int) -> Tuple[str, bool]:
    # This should be implemented externally
    import time
    time.sleep(0.1)
    import random
    next_token = random.choice([" Yes", " No", " Maybe", " Perhaps", " Probably"])
    done = random.choices([True, False], weights=[0.1, 0.9], k=1)[0]
    return (prompt + next_token, done)

class ChatSession:
    def __init__(self, name: str):
        self.name = name
        self.history: List[Tuple[str, str]] = []  # list of (user, bot) turns

class ChatApp(App):
    CSS_PATH = None
    BINDINGS = [("n", "new_chat", "New Chat"), ("q", "quit", "Quit")]

    current_chat: reactive[str | None] = reactive(None)
    sessions: reactive[Dict[str, ChatSession]] = reactive({})

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):  # Left panel
                yield Button("New Chat", id="new_chat_btn")
                yield ListView(id="chat_list")
                yield Static("Model:")
                yield Select([("meta-llama/LLama-3.2-3B", "meta-llama/LLama-3.2-3B")], id="model_select")
                yield Static("k:")
                yield Input(value="1", placeholder="Enter k", id="k_select")
            with Vertical(id="main_panel"):  # Right panel
                yield Static("No chat selected", id="chat_title")
                yield Static("", id="chat_view", expand=True)
                yield Input(placeholder="Type your message and press Enter", id="chat_input")
        yield Footer()

    async def on_mount(self):
        # initialize with one chat
        await self.action_new_chat()

    async def action_new_chat(self):
        name = f"Chat {len(self.sessions) + 1}"
        session = ChatSession(name)
        self.sessions[name] = session
        list_view: ListView = self.query_one("#chat_list")
        safe_id = f"chat_{name.replace(' ', '_')}"
        list_view.append(ListItem(Button(name, id=safe_id)))
        self.current_chat = name
        await self.refresh_chat()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "new_chat_btn":
            await self.action_new_chat()
        elif btn_id.startswith("chat_"):
            name = btn_id.replace("chat_", "").replace("_", " ")
            self.current_chat = name
            await self.refresh_chat()

    async def refresh_chat(self):
        title: Static = self.query_one("#chat_title")
        chat_view: Static = self.query_one("#chat_view")
        session = self.sessions.get(self.current_chat)
        if not session:
            title.update("No chat selected")
            chat_view.update("")
            return
        title.update(session.name)
        # render history
        chat_text = "".join(f"You: {u}\nBot: {b}\n\n" for u, b in session.history)
        chat_view.update(chat_text)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not self.current_chat:
            return
        user_msg = event.value.strip()
        event.input.value = ""
        session = self.sessions[self.current_chat]
        session.history.append((user_msg, ""))
        await self.refresh_chat()
        # start streaming response
        model = self.query_one("#model_select", Select).value
        try:
            k = int(self.query_one("#k_select", Input).value)
        except ValueError:
            k = 1

        # generate tokens
        bot_text = ""
        while True:
            bot_text, done = await asyncio.to_thread(llm_pipeline, f"{bot_text}", model, k)
            session.history[-1] = (user_msg, bot_text)
            await self.refresh_chat()
            if done:
                break

    def action_quit(self) -> None:
        self.exit()

if __name__ == "__main__":
    ChatApp().run()
