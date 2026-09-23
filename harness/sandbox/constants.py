"""Container lifecycle settings shared by the host-side sandbox wrapper."""

from pathlib import Path

DOCKER_EXECUTABLE = "docker"
CONTAINER_NAME_PREFIX = "harness-"
CONTAINER_RUN_DIRECTORY = "/run/attempt"
ROOT_USER_ID = 0
CONTROL_TIMEOUT_SECONDS = 30
SUCCESS_EXIT_CODE = 0
UNRESTRICTED_NETWORKS = {"host", "bridge", "default"}
PROVIDER_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
PROVIDER_PORTS = {"http": 80, "https": 443}
RUN_UID_ENV = "HARNESS_RUN_UID"
RUN_GID_ENV = "HARNESS_RUN_GID"
BOOTSTRAP_MODULE = "harness.sandbox.bootstrap"
BOOTSTRAP_CAPABILITIES = ("NET_ADMIN", "SETUID", "SETGID", "SETPCAP")
PYTHON_EXECUTABLE = "python"
PRIVILEGE_EXECUTABLE = "setpriv"
HOSTS_FILE = Path("/etc/hosts")
MIN_PROVIDER_PORT = 1
FIREWALL_EXECUTABLES = {4: "iptables", 6: "ip6tables"}
COMMAND_ARGUMENT_OFFSET = 1
MISSING_CONTAINER_ERROR_PREFIX = "Error response from daemon: No such container: "
RUN_DIRECTORY_LABEL = "harness.run-directory"
