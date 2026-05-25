"""Slack connector — index recent messages from a channel.

Needs SLACK_BOT_TOKEN. The bot must have `channels:history` (and `groups:history`
for private channels), and must be invited to the channel you want to read.
"""
from __future__ import annotations

import os
from typing import Iterator

import httpx

from . import connectors
from .base import Connector, Document


@connectors.register("slack")
class SlackConnector(Connector):
    name = "Slack channel"
    description = "Index recent messages from a Slack channel. Needs SLACK_BOT_TOKEN."
    input_kind = "text"
    placeholder = "Channel ID (e.g. C0123456789)"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.getenv("SLACK_BOT_TOKEN", "").strip())

    def __init__(self) -> None:
        self.token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        self.max_messages = int(os.getenv("SLACK_MAX_MESSAGES", "300"))

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        channel_id = payload.strip()
        if not channel_id:
            raise ValueError("Channel ID is empty.")

        headers = {"Authorization": f"Bearer {self.token}"}
        with httpx.Client(headers=headers, timeout=30.0) as client:
            r = client.get(
                "https://slack.com/api/conversations.history",
                params={"channel": channel_id, "limit": self.max_messages},
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                raise RuntimeError(
                    f"Slack API error: {data.get('error', 'unknown')}"
                )
            messages = data.get("messages") or []

        # Oldest first so chronology reads naturally.
        for msg in reversed(messages):
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            user = msg.get("user", "?")
            ts = msg.get("ts", "")
            yield Document(
                text=f"<@{user}>: {text}",
                source=f"slack:{channel_id}#{ts}",
            )
