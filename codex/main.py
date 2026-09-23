"""Small host CLI for private Codex subscription gateways."""

import asyncio
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from subprocess import CalledProcessError

from aiohttp import ClientError, ClientSession, ClientTimeout
from dotenv import load_dotenv
from harness.sandbox.constants import COMMAND_ARGUMENT_OFFSET, SUCCESS_EXIT_CODE
from harness.sandbox.docker import Docker
from harness.utils.asyncio import run_process
from harness.utils.constants import TEXT_ENCODING
from launcher.constants import (
    DOTENV_FILE,
    FAILED_EXIT_CODE,
    LITELLM_KEY_ENV,
    LITELLM_URL_ENV,
    LOG_FORMAT,
)

from codex.constants import HEALTH_TIMEOUT_SECONDS
from codex.gateway import Gateway
from codex.models import Account


def parse_arguments(arguments: list[str]) -> Namespace:
    """Accept explicit account arrays and only the settings needed by each command."""
    parser = ArgumentParser(prog="python -m codex.main", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", allow_abbrev=False)
    add = commands.add_parser("add", allow_abbrev=False)
    add.add_argument("--account", required=True, type=int)
    add.add_argument("--auth-file", required=True, type=Path)
    for name in ("start", "check", "stop", "env"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--accounts", required=True, type=int, nargs="+")
    args = parser.parse_args(arguments)
    if args.command == "add":
        Account(args.account)
    elif args.command != "build":
        args.accounts = [Account(number) for number in args.accounts]
        if len(set(args.accounts)) != len(args.accounts):
            parser.error("accounts must be unique")
    return args


async def execute_command(args: Namespace) -> int:
    """Compose injected infrastructure and keep account setup separate from experiment launch."""
    async with ClientSession(timeout=ClientTimeout(total=HEALTH_TIMEOUT_SECONDS), trust_env=False) as client:
        gateway = Gateway(Docker(asyncio.create_subprocess_exec), client, asyncio.sleep)
        if args.command == "build":
            await gateway.build()
        elif args.command == "add":
            await gateway.add(Account(args.account), args.auth_file)
        elif args.command == "start":
            await gateway.start(args.accounts, _read_gateway_key())
        else:
            for account in args.accounts:
                if args.command == "check":
                    await gateway.check(account, _read_gateway_key())
                    print(f"account={account.number} ready url={account.url}")
                elif args.command == "stop":
                    await gateway.stop(account)
                else:
                    print(f"{LITELLM_URL_ENV}_{account.number}={account.url}")
    return SUCCESS_EXIT_CODE


def _read_gateway_key() -> str:
    """Require an explicit local gateway key only for authenticated operations."""
    key = os.environ[LITELLM_KEY_ENV].strip()
    if not key or any(value in key for value in "\r\n"):
        raise ValueError(f"{LITELLM_KEY_ENV} must be a nonempty single-line gateway key")
    return key


def run_gateway() -> None:
    """Load host settings and report failures without exposing credential-bearing subprocess input."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    load_dotenv(DOTENV_FILE, override=False)
    try:
        code = run_process(execute_command(parse_arguments(sys.argv[COMMAND_ARGUMENT_OFFSET:])))
    except CalledProcessError as error:
        logging.error("Codex gateway Docker command failed (exit %s): %s", error.returncode, error.stderr.decode(TEXT_ENCODING))
        code = FAILED_EXIT_CODE
    except (ValueError, KeyError, OSError, ClientError, RuntimeError) as error:
        logging.error("Codex gateway failed: %s", error)
        code = FAILED_EXIT_CODE
    raise SystemExit(code)


if __name__ == "__main__":
    run_gateway()
