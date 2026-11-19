"""Thread-safe, credential-free execution trace for frozen benchmark runs."""

import threading
from time import monotonic

from langchain_core.callbacks import BaseCallbackHandler


class EvaluationTrace(BaseCallbackHandler):
    def __init__(self, *, capture_content=True):
        self.lock = threading.Lock()
        self.capture_content = capture_content
        self.events = []
        self.started = {}
        self.tokens_in = 0
        self.tokens_out = 0
        self.usage_calls = 0
        self.llm_calls = 0

    def event(self, kind, **fields):
        with self.lock:
            self.events.append({"kind": kind, **fields})

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.lock:
            self.started[str(run_id)] = monotonic()
            self.llm_calls += 1
            event = {"kind": "llm_start", "id": str(run_id)}
            if self.capture_content:
                event["messages"] = [[str(m.content) for m in batch] for batch in messages]
            self.events.append(event)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        with self.lock:
            self.started[str(run_id)] = monotonic()
            self.llm_calls += 1
            event = {"kind": "llm_start", "id": str(run_id)}
            if self.capture_content:
                event["prompts"] = prompts
            self.events.append(event)

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self.lock:
            generations = [item for batch in response.generations for item in batch]
            for item in generations:
                usage = getattr(getattr(item, "message", None), "usage_metadata", None)
                if usage:
                    self.tokens_in += usage.get("input_tokens", 0)
                    self.tokens_out += usage.get("output_tokens", 0)
                    self.usage_calls += 1
            started = self.started.pop(str(run_id), None)
            event = {"kind": "llm_end", "id": str(run_id),
                     "seconds": monotonic() - started if started is not None else None}
            if self.capture_content:
                event["output"] = [getattr(item, "text", "") for item in generations]
            self.events.append(event)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.event("llm_error", id=str(run_id), error_type=type(error).__name__)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        fields = {"id": str(run_id), "name": serialized.get("name")}
        if self.capture_content:
            fields["arguments"] = input_str
        self.event("tool_start", **fields)

    def on_tool_end(self, output, *, run_id, **kwargs):
        fields = {"id": str(run_id)}
        if self.capture_content:
            fields["output"] = str(getattr(output, "content", output))
        self.event("tool_end", **fields)

    def summary(self):
        with self.lock:
            return {"llm_calls": self.llm_calls, "input_tokens": self.tokens_in,
                    "output_tokens": self.tokens_out, "calls_with_usage": self.usage_calls,
                    "token_usage_complete": self.llm_calls > 0 and self.llm_calls == self.usage_calls,
                    "tool_calls": sum(e["kind"] == "frozen_tool" for e in self.events)}
