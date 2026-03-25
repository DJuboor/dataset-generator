"""Multi-turn conversation dataset generation."""

from __future__ import annotations

import json
import logging

from dataset_generator.tasks.base import Sample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a dataset generation assistant. Generate realistic multi-turn \
conversations between a user and an AI assistant. Each conversation should flow naturally, \
with the user asking follow-up questions and the assistant providing helpful responses.

Respond with a JSON array of objects, each with a "messages" field containing an array of \
message objects with "role" (user/assistant) and "content" fields.
Do NOT include any text outside the JSON array."""

USER_PROMPT_TEMPLATE = """Generate {batch_size} diverse multi-turn conversations.

{domain_context}
{system_prompt_context}
Turns per conversation: {min_turns} to {max_turns} (total messages, alternating user/assistant)

Requirements:
- Each conversation must start with a user message
- Messages must alternate between user and assistant roles
- Conversations should feel natural with follow-ups, clarifications, and topic evolution
- Vary conversation styles: informational, troubleshooting, creative, analytical
- Assistant responses should be helpful and contextually aware of prior turns
- Do NOT generate trivially short or repetitive exchanges

Respond with a JSON array:
[{{"messages": [{{"role": "user", "content": "How do I..."}}, \
{{"role": "assistant", "content": "You can..."}}]}}]"""


class ConversationTask:
    """Generate multi-turn conversation datasets for chat model training."""

    def __init__(
        self,
        domain: str = "",
        min_turns: int = 2,
        max_turns: int = 6,
        system_prompt: str = "",
    ):
        self.domain = domain
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.system_prompt = system_prompt

    @classmethod
    def from_config(cls, config: dict) -> ConversationTask:
        """Create from config dict."""
        task_config = config.get("task", config)
        return cls(
            domain=task_config.get("domain", ""),
            min_turns=task_config.get("min_turns", 2),
            max_turns=task_config.get("max_turns", 6),
            system_prompt=task_config.get("system_prompt", ""),
        )

    def build_messages(self, batch_size: int = 3) -> list[dict[str, str]]:
        """Build messages for generating multi-turn conversations."""
        domain_context = f"Domain: {self.domain}" if self.domain else ""
        system_prompt_context = (
            f"System prompt for the assistant in conversations: {self.system_prompt}"
            if self.system_prompt
            else ""
        )

        user_msg = USER_PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            domain_context=domain_context,
            system_prompt_context=system_prompt_context,
            min_turns=self.min_turns,
            max_turns=self.max_turns,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def parse_response(self, response: str) -> list[Sample]:
        """Parse JSON array of multi-turn conversations."""
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse conversation response as JSON")
            return []

        samples = []
        for item in items:
            if not isinstance(item, dict) or "messages" not in item:
                continue
            messages = item["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                logger.debug("Skipping conversation with fewer than 2 messages")
                continue
            # Validate alternating roles starting with user
            valid = True
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    valid = False
                    break
                expected_role = "user" if i % 2 == 0 else "assistant"
                if msg["role"] != expected_role:
                    valid = False
                    break
            if not valid:
                logger.debug("Skipping conversation with invalid role alternation")
                continue

            first_user_message = messages[0]["content"]
            samples.append(
                Sample(
                    text=first_user_message,
                    metadata={
                        "messages": messages,
                        "system_prompt": self.system_prompt,
                    },
                )
            )
        return samples
