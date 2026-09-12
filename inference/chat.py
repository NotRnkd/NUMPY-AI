"""Interactive chat session manager with system prompt preservation and context truncation."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

from inference.generate import generate_stream
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer


class ChatSession:
    def __init__(
        self,
        model: GPT,
        tokenizer: ByteBPETokenizer,
        system_prompt: str = "You are NumPy-GPT, a helpful, honest, and offline AI language model running locally on NumPy.",
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 200,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.messages: List[Dict[str, str]] = []
        self.reset()

    def reset(self) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def build_prompt(self, user_query: str) -> str:
        """Construct prompt and truncate middle conversation turns if context exceeds limit."""
        tok = self.tokenizer
        max_ctx = self.model.config.context_length - self.max_new_tokens - 10

        # Tentative messages including new user turn
        candidate_turns = list(self.messages) + [{"role": "user", "content": user_query}]

        def format_conversation(msgs: List[Dict[str, str]]) -> str:
            parts = []
            for m in msgs:
                r = m["role"]
                c = m["content"]
                if r == "system":
                    parts.append(f"<system>{c}")
                elif r == "user":
                    parts.append(f"<user>{c}")
                elif r == "assistant":
                    parts.append(f"<assistant>{c}<eos>")
            parts.append("<assistant>")
            return "".join(parts)

        prompt_str = format_conversation(candidate_turns)
        encoded = tok.encode(prompt_str, allowed_special=True)

        # If it fits within context budget, return directly
        if len(encoded) <= max_ctx:
            return prompt_str

        # Truncation strategy: preserve system prompt (index 0) and remove oldest (user, assistant) pairs
        system_turn = candidate_turns[0]
        recent_turns = candidate_turns[1:]

        while len(recent_turns) > 1 and len(encoded) > max_ctx:
            # Drop the oldest turn
            recent_turns.pop(0)
            prompt_str = format_conversation([system_turn] + recent_turns)
            encoded = tok.encode(prompt_str, allowed_special=True)

        return prompt_str

    def respond(self, user_query: str, stream: bool = True) -> str:
        prompt = self.build_prompt(user_query)

        # Stream or collect
        pieces = []
        stream_gen = generate_stream(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            use_cache=True,
        )

        for chunk in stream_gen:
            pieces.append(chunk)
            if stream:
                print(chunk, end="", flush=True)

        full_response = "".join(pieces).strip()
        if stream:
            print()

        # Update conversation history
        self.messages.append({"role": "user", "content": user_query})
        self.messages.append({"role": "assistant", "content": full_response})
        return full_response


def interactive_chat_repl(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> None:
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt or "You are NumPy-GPT, an intelligent and helpful offline AI assistant.",
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    print("=" * 60)
    print("                NumPy-GPT Interactive Chat")
    print("=" * 60)
    print(f"Model parameters: {model.parameter_count():,} | Context window: {model.config.context_length}")
    print("Type your message and press Enter.")
    print("Commands: /reset (clear history), /system <prompt>, /help, /exit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            break
        elif user_input.lower() == "/reset":
            session.reset()
            print("[History cleared]")
            continue
        elif user_input.lower() == "/help":
            print("Commands:")
            print("  /reset          - Clear conversation history")
            print("  /system <text>  - Set a new system prompt")
            print("  /history        - View past turns")
            print("  /exit           - Exit chat")
            continue
        elif user_input.lower().startswith("/system "):
            new_sys = user_input[8:].strip()
            session.system_prompt = new_sys
            session.reset()
            print(f"[System prompt updated to: {new_sys}]")
            continue
        elif user_input.lower() == "/history":
            print("\n--- History ---")
            for m in session.messages:
                print(f"[{m['role'].upper()}]: {m['content']}")
            print("---------------")
            continue

        print("AI: ", end="", flush=True)
        session.respond(user_input, stream=True)
