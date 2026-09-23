"""Execute one isolated judge task with validated output and bounded connection recovery."""

import json
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any, ClassVar

from claude_agent_sdk import ClaudeAgentOptions
from harness.session.agent_session import AgentSession
from harness.session.connection_manager import ConnectionManager
from harness.session.constants import (
    LOCAL_TOOLS,
    STRUCTURED_OUTPUT_TOOL,
)
from harness.session.models import EventKind
from harness.utils.constants import (
    COUNT_INCREMENT,
    DEFAULT_AUDIT_MAX_TURNS,
    INITIAL_COUNT,
    RECOVERY_MAX_RETRIES,
    TEXT_ENCODING,
)
from harness.utils.prompt_loader import load_prompt
from harness.utils.retry import wait_before_retry
from harness.utils.storage import atomic_write, fresh_runtime
from jsonschema import Draft202012Validator

from audit.constants import AUDIT_SYSTEM_PROMPT, RESULT_FILENAME
from audit.models import AuditRequest


class BaseAudit:
    """Run and persist a single judgment; subclasses declare name, prompt_file, and schema.

    Override _get_prompt_values only for additional prompt inputs. Inject connection creation and retry waiting; read_result validates the temporary worker output before host publication.
    """

    name: ClassVar[str]
    prompt_file: ClassVar[Path]
    schema: ClassVar[dict[str, Any]]

    def __init__(self, connections: Callable[[ClaudeAgentOptions], ConnectionManager], wait: Callable[[float], Awaitable[None]]) -> None:
        """Accept the existing connection transport without introducing refinement state."""
        self._connections = connections
        self._wait = wait

    async def run(self, request: AuditRequest, directory: Path) -> None:
        """Retry only transient connections, then atomically save a validated verdict after cleanup."""
        if request.kind != self.name:
            raise ValueError("request does not match this audit")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RESULT_FILENAME).unlink(missing_ok=True)
        prompt = load_prompt(self.prompt_file, self._get_prompt_values(request))
        retries = INITIAL_COUNT
        while True:
            try:
                verdict = await self._judge(prompt, request.model, directory)
                break
            except (ConnectionError, TimeoutError) as error:
                if retries == RECOVERY_MAX_RETRIES:
                    raise
                await wait_before_retry(error, retries, self._wait)
                retries += COUNT_INCREMENT
        content = json.dumps(verdict, ensure_ascii=False).encode(TEXT_ENCODING)
        atomic_write(directory / RESULT_FILENAME, lambda handle: handle.write(content))

    @classmethod
    def read_result(cls, directory: Path) -> dict[str, Any]:
        """Read the worker's validated verdict; missing or corrupt output fails loudly."""
        return cls._validate_result(json.loads((directory / RESULT_FILENAME).read_text(encoding=TEXT_ENCODING)))

    @classmethod
    def _validate_result(cls, value: Any) -> dict[str, Any]:
        """Require the exact output schema before publishing or accepting a saved result."""
        Draft202012Validator(cls.schema).validate(value)
        return value

    def _get_prompt_values(self, request: AuditRequest) -> dict[str, str]:
        """Provide the mathematical inputs shared by both audit prompts."""
        return {"statement": request.problem, "reference_solution": request.reference, "solution": request.solution}

    async def _judge(self, prompt: str, model: str, directory: Path) -> dict[str, Any]:
        """Own a fresh scratch directory and connection for this one bounded judging attempt."""
        options = ClaudeAgentOptions(model=model, system_prompt=AUDIT_SYSTEM_PROMPT,
                                    allowed_tools=[*LOCAL_TOOLS, STRUCTURED_OUTPUT_TOOL], max_turns=DEFAULT_AUDIT_MAX_TURNS,
                                    output_format={"type": "json_schema", "schema": self.schema})
        connection = self._connections(options)
        with fresh_runtime(directory) as runtime:
            try:
                await connection.connect(runtime, None, False, None)
                verdict = await self._read_verdict(AgentSession(connection, set()), prompt)
            except BaseException as error:
                await connection.close_after_failure(error)
                raise
            else:
                await connection.close()
                return verdict

    async def _read_verdict(self, session: AgentSession, prompt: str) -> dict[str, Any]:
        """Consume normal tool-capable execution and accept only a completed, valid JSON verdict."""
        async with aclosing(session.run(prompt)) as events:
            async for event in events:
                if event.kind == EventKind.COMPLETED:
                    return self._validate_result(json.loads(event.text))
                if event.kind == EventKind.INTERRUPTED:
                    raise RuntimeError("audit interrupted before a verdict")
        raise ConnectionError("audit ended without a verdict")
