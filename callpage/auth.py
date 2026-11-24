"""
auth.py -- accounts, sessions and "who owns what" for the call server (standard library only)
=============================================================================================

* ``UserStore``    data/users.json: user name, display name, salted PBKDF2 password hash, role (admin | user).
                   Anyone can create a caller account (name + user name + password); there is NO e-mail and no code.
                   The administrator account is created ONCE, on the admin page, while none exists (or from
                   SVARA_ADMIN_USERNAME / SVARA_ADMIN_PASSWORD in .env).  Passwords are never stored or logged, and a
                   wrong password is slowed down (5 failures -> 60 s lock per user name).
* ``SessionStore`` random bearer tokens kept in memory (a server restart signs everybody out).
* ``OwnerStore``   data/owners.json: which user owns a private form, an uploaded template or a call.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from typing import Any, Dict, Optional

log = logging.getLogger("callpage.auth")
_USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{2,29}$")
_ITER = 200_000
SESSION_TTL_S = 12 * 3600


class AuthError(Exception):
    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITER).hex()


def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class UserStore:
    def __init__(self, path: str, clock=time.time):
        self.path, self.clock = path, clock
        self._lock = threading.Lock()
        self._fails: Dict[str, list] = {}
        self._users: Dict[str, Dict[str, Any]] = {}
        self.renamed: Dict[str, str] = {}              # old owner key (e-mail) -> user name, after a migration
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self._users = json.load(fh)
            except (OSError, ValueError):
                self._users = {}
        self._migrate()
        self._seed_admin()

    def _save(self) -> None:
        _atomic_write(self.path, self._users)

    def _migrate(self) -> None:
        """Older versions keyed accounts by e-mail and needed a confirmation code. Re-key by user name; drop accounts that
        never finished confirming and the old built-in administrator (its default password was public)."""
        if not any("@" in k or "verified" in v for k, v in self._users.items()):
            return
        fresh: Dict[str, Dict[str, Any]] = {}
        for key, u in self._users.items():
            username = (u.get("username") or key.split("@")[0]).lower()
            if u.get("verified") is False:
                continue
            if u.get("role") == "admin" and key == "admin@svara.local" and \
                    hmac.compare_digest(u.get("hash", ""), _hash("svara-admin", bytes.fromhex(u.get("salt", "00")))):
                log.warning("removed the old built-in administrator account (default password)")
                continue
            if username in fresh:
                continue
            if "@" in key:
                self.renamed[key] = username
            fresh[username] = {"username": username, "name": u.get("name") or username, "role": u.get("role", "user"),
                               "salt": u["salt"], "hash": u["hash"], "created": u.get("created", int(self.clock()))}
        self._users = fresh
        self._save()

    def _seed_admin(self) -> None:
        username = (os.environ.get("SVARA_ADMIN_USERNAME") or "").strip().lower()
        password = os.environ.get("SVARA_ADMIN_PASSWORD") or ""
        if not username or not password or self.admin_exists() or not _USERNAME.match(username):
            return
        self._users[username] = self._new(username, os.environ.get("SVARA_ADMIN_NAME") or "Administrator", password, "admin")
        self._save()

    def _new(self, username: str, name: str, password: str, role: str) -> Dict[str, Any]:
        salt = secrets.token_bytes(16)
        return {"username": username, "name": name, "role": role, "salt": salt.hex(), "hash": _hash(password, salt),
                "created": int(self.clock())}

    def admin_exists(self) -> bool:
        return any(u.get("role") == "admin" for u in self._users.values())

    @staticmethod
    def public(user: Dict[str, Any]) -> Dict[str, Any]:
        return {"username": user["username"], "name": user["name"], "role": user["role"]}

    def signup(self, name: str, username: str, password: str, role: str = "user") -> Dict[str, Any]:
        name, username = (name or "").strip(), (username or "").strip().lower()
        if len(name) < 2 or len(name) > 60:
            raise AuthError(400, "Please enter your name.")
        if not _USERNAME.match(username):
            raise AuthError(400, "A user name needs 3 to 30 letters, digits, dots, dashes or underscores (no @ or spaces), for example athira.admin.")
        if not isinstance(password, str) or len(password) < 6 or len(password) > 128:
            raise AuthError(400, "Choose a password of at least 6 characters.")
        with self._lock:
            if role == "admin" and self.admin_exists():
                raise AuthError(403, "An administrator account already exists. Only the existing administrator can add another.")
            if username in self._users:
                raise AuthError(409, "That user name is taken. Please choose another.")
            user = self._new(username, name, password, "admin" if role == "admin" else "user")
            self._users[username] = user
            self._save()
            return self.public(user)

    def login(self, username: str, password: str, role: Optional[str] = None) -> Dict[str, Any]:
        key = (username or "").strip().lower()
        now = self.clock()
        with self._lock:
            fails = [t for t in self._fails.get(key, []) if now - t < 60]
            if len(fails) >= 5:
                raise AuthError(429, "Too many attempts. Please wait a minute and try again.")
            user = self._users.get(key)
            ok = False
            if user and isinstance(password, str):
                ok = hmac.compare_digest(user["hash"], _hash(password, bytes.fromhex(user["salt"])))
            elif isinstance(password, str):
                _hash(password, b"0" * 16)                                           # same time whether or not the account exists
            if not ok or (role and user["role"] != role):
                fails.append(now)
                self._fails[key] = fails
                raise AuthError(401, "That user name or password is not right.")
            self._fails.pop(key, None)
            return self.public(user)


class SessionStore:
    def __init__(self, clock=time.time):
        self.clock = clock
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, user: Dict[str, Any]) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._items = {k: v for k, v in self._items.items() if v["exp"] > self.clock()}
            self._items[token] = {**user, "exp": self.clock() + SESSION_TTL_S}
        return token

    def get(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self._lock:
            item = self._items.get(token)
            if item and item["exp"] > self.clock():
                return {k: v for k, v in item.items() if k != "exp"}
            self._items.pop(token, None)
        return None

    def drop(self, token: Optional[str]) -> None:
        with self._lock:
            self._items.pop(token or "", None)


class OwnerStore:
    """Maps "form:<id>", "tpl:<id>" and "call:<id>" to the user name of whoever created it (absent = shared / admin)."""

    def __init__(self, path: str):
        self.path, self._lock, self._d = path, threading.Lock(), {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self._d = json.load(fh)
            except (OSError, ValueError):
                self._d = {}

    def get(self, kind: str, ident: str) -> Optional[str]:
        return self._d.get(f"{kind}:{ident}")

    def set(self, kind: str, ident: str, owner: str) -> None:
        with self._lock:
            self._d[f"{kind}:{ident}"] = owner
            _atomic_write(self.path, self._d)

    def drop(self, kind: str, ident: str) -> None:
        with self._lock:
            if self._d.pop(f"{kind}:{ident}", None) is not None:
                _atomic_write(self.path, self._d)

    def rename_owners(self, mapping: Dict[str, str]) -> None:
        if not mapping:
            return
        with self._lock:
            self._d = {k: mapping.get(v, v) for k, v in self._d.items()}
            _atomic_write(self.path, self._d)


class AccessStore:
    """Per-form access codes (data/access.json). A caller can open a form that has a code only after entering it once;
    changing or removing the code revokes everybody's earlier unlock.  Callers never receive the code, only the admin does."""

    ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"                       # no 0/O, 1/I/L: easy to read out loud

    def __init__(self, path: str, clock=time.time):
        self.path, self.clock, self._lock = path, clock, threading.Lock()
        self._fails: Dict[str, list] = {}
        self._d: Dict[str, Any] = {"codes": {}, "unlocks": {}}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                self._d = {"codes": loaded.get("codes", {}), "unlocks": loaded.get("unlocks", {})}
            except (OSError, ValueError):
                pass

    def _save(self) -> None:
        _atomic_write(self.path, self._d)

    @staticmethod
    def normalize(code: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", code or "").upper()

    def generate(self) -> str:
        return "".join(secrets.choice(self.ALPHABET) for _ in range(6))

    def code_of(self, form_id: str) -> Optional[str]:
        item = self._d["codes"].get(form_id)
        return item["code"] if item else None

    def set_code(self, form_id: str, code: Optional[str]) -> Optional[str]:
        """``None`` removes the code (the form becomes open). A new code makes everybody enter it again."""
        with self._lock:
            if code is None:
                self._d["codes"].pop(form_id, None)
            else:
                code = self.normalize(code)
                if not 4 <= len(code) <= 16:
                    raise AuthError(400, "An access code needs 4 to 16 letters or digits.")
                old = self._d["codes"].get(form_id)
                self._d["codes"][form_id] = {"code": code, "version": (old["version"] + 1) if old else 1}
            self._save()
            return code

    def is_unlocked(self, username: str, form_id: str) -> bool:
        item = self._d["codes"].get(form_id)
        if not item:
            return True
        return self._d["unlocks"].get(username, {}).get(form_id) == item["version"]

    def unlock(self, username: str, form_id: str, code: str) -> None:
        key, now = f"{username}:{form_id}", self.clock()
        with self._lock:
            item = self._d["codes"].get(form_id)
            if not item:
                return
            fails = [t for t in self._fails.get(key, []) if now - t < 60]
            if len(fails) >= 5:
                raise AuthError(429, "Too many wrong codes. Please wait a minute and try again.")
            if not hmac.compare_digest(item["code"].encode(), self.normalize(code).encode()):
                fails.append(now)
                self._fails[key] = fails
                raise AuthError(403, "That access code is not right.", code_wrong=True)
            self._fails.pop(key, None)
            self._d["unlocks"].setdefault(username, {})[form_id] = item["version"]
            self._save()

    def unlock_by_code(self, username: str, code: str) -> list:
        """Unlock every form whose code matches (the caller does not say which form). Wrong guesses are rate limited."""
        key, now = f"{username}:*", self.clock()
        with self._lock:
            fails = [t for t in self._fails.get(key, []) if now - t < 60]
            if len(fails) >= 5:
                raise AuthError(429, "Too many wrong codes. Please wait a minute and try again.")
            wanted = self.normalize(code).encode()
            hits = [(fid, item) for fid, item in self._d["codes"].items() if hmac.compare_digest(item["code"].encode(), wanted)]
            if not hits or not wanted:
                fails.append(now)
                self._fails[key] = fails
                raise AuthError(403, "That access code is not right.", code_wrong=True)
            self._fails.pop(key, None)
            for fid, item in hits:
                self._d["unlocks"].setdefault(username, {})[fid] = item["version"]
            self._save()
            return [fid for fid, _ in hits]

    def forget(self, form_id: str) -> None:
        with self._lock:
            self._d["codes"].pop(form_id, None)
            for forms in self._d["unlocks"].values():
                forms.pop(form_id, None)
            self._save()


class Auth:
    """Bundle used by ``CallApp``: users + sessions + owners."""

    def __init__(self, data_dir: str):
        self.users = UserStore(os.path.join(data_dir, "users.json"))
        self.sessions = SessionStore()
        self.owners = OwnerStore(os.path.join(data_dir, "owners.json"))
        self.owners.rename_owners(self.users.renamed)
        self.access = AccessStore(os.path.join(data_dir, "access.json"))
