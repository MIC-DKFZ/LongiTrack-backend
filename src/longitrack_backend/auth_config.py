"""Fixed server-side authorization storage settings."""

from pathlib import Path

AUTHORIZED_KEYS_DIR = Path("/srv/longitrack/authorized_keys")
REMOTE_AUTHORIZED_KEYS_DIR = "/srv/longitrack/authorized_keys"
AUTHORIZED_KEYS_GROUP = "longitrack"
