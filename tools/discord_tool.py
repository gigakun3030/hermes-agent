"""Discord server introspection and management tool.

Provides the agent with the ability to interact with Discord servers
when running on the Discord gateway. Uses Discord REST API directly
with the bot token — no dependency on the gateway adapter's client.

Only included in the hermes-discord toolset, so it has zero cost
for users on other platforms.

The schema exposed to the model is filtered by two gates:

1. Privileged intents detected from GET /applications/@me at schema
   build time. Actions that require an intent the bot doesn't have
   (search_members / member_info → GUILD_MEMBERS intent) are hidden.
   fetch_messages is kept regardless of MESSAGE_CONTENT intent, but
   its description is annotated when the intent is missing.

2. User config allowlist at ``discord.server_actions``. If the user
   sets a comma-separated list (or YAML list) of action names, only
   those appear in the schema. Empty/unset means all intent-available
   actions are exposed.

Per-guild permissions (MANAGE_ROLES etc.) are NOT pre-checked — Discord
returns a 403 at call time and :func:`_enrich_403` maps it to
actionable guidance the model can relay to the user.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"

# Application flag bits (from GET /applications/@me → "flags").
# Source: https://discord.com/developers/docs/resources/application#application-object-application-flags
_FLAG_GATEWAY_GUILD_MEMBERS = 1 << 14
_FLAG_GATEWAY_GUILD_MEMBERS_LIMITED = 1 << 15
_FLAG_GATEWAY_MESSAGE_CONTENT = 1 << 18
_FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED = 1 << 19

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bot_token() -> Optional[str]:
    """Resolve the Discord bot token from environment."""
    return os.getenv("DISCORD_BOT_TOKEN", "").strip() or None


def _discord_request(
    method: str,
    path: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Any:
    """Make a request to the Discord REST API."""
    url = f"{DISCORD_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent (https://github.com/NousResearch/hermes-agent)",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise DiscordAPIError(e.code, error_body) from e


class DiscordAPIError(Exception):
    """Raised when a Discord API call fails."""
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Discord API error {status}: {body}")


# ---------------------------------------------------------------------------
# Channel type mapping
# ---------------------------------------------------------------------------

_CHANNEL_TYPE_NAMES = {
    0: "text",
    2: "voice",
    4: "category",
    5: "announcement",
    10: "announcement_thread",
    11: "public_thread",
    12: "private_thread",
    13: "stage",
    15: "forum",
    16: "media",
}


def _channel_type_name(type_id: int) -> str:
    return _CHANNEL_TYPE_NAMES.get(type_id, f"unknown({type_id})")


# ---------------------------------------------------------------------------
# Capability detection (application intents)
# ---------------------------------------------------------------------------

# Module-level cache so the app/me endpoint is hit at most once per process.
_capability_cache: Dict[str, Dict[str, Any]] = {}


def _detect_capabilities(token: str, *, force: bool = False) -> Dict[str, Any]:
    """Detect the bot's app-wide capabilities via GET /applications/@me.

    Returns a dict with keys:

    - ``has_members_intent``: GUILD_MEMBERS intent is enabled
    - ``has_message_content``: MESSAGE_CONTENT intent is enabled
    - ``detected``: detection succeeded (False means exposing everything
      and letting runtime errors handle it)

    Cached in a module-global. Pass ``force=True`` to re-fetch.
    """
    global _capability_cache
    if token in _capability_cache and not force:
        return _capability_cache[token]

    caps: Dict[str, Any] = {
        "has_members_intent": True,
        "has_message_content": True,
        "detected": False,
    }

    try:
        app = _discord_request("GET", "/applications/@me", token, timeout=5)
        flags = int(app.get("flags", 0) or 0)
        caps["has_members_intent"] = bool(
            flags & (_FLAG_GATEWAY_GUILD_MEMBERS | _FLAG_GATEWAY_GUILD_MEMBERS_LIMITED)
        )
        caps["has_message_content"] = bool(
            flags & (_FLAG_GATEWAY_MESSAGE_CONTENT | _FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED)
        )
        caps["detected"] = True
    except Exception as exc:  # nosec — detection is best-effort
        logger.info(
            "Discord capability detection failed (%s); exposing all actions.", exc,
        )

    _capability_cache[token] = caps
    return caps


def _reset_capability_cache() -> None:
    """Test hook: clear the detection cache."""
    global _capability_cache
    _capability_cache = {}


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _list_guilds(token: str, **_kwargs: Any) -> str:
    """List all guilds the bot is a member of."""
    guilds = _discord_request("GET", "/users/@me/guilds", token)
    result = []
    for g in guilds:
        result.append({
            "id": g["id"],
            "name": g["name"],
            "icon": g.get("icon"),
            "owner": g.get("owner", False),
            "permissions": g.get("permissions"),
        })
    return json.dumps({"guilds": result, "count": len(result)})


def _server_info(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get detailed information about a guild."""
    g = _discord_request("GET", f"/guilds/{guild_id}", token, params={"with_counts": "true"})
    return json.dumps({
        "id": g["id"],
        "name": g["name"],
        "description": g.get("description"),
        "icon": g.get("icon"),
        "owner_id": g.get("owner_id"),
        "member_count": g.get("approximate_member_count"),
        "online_count": g.get("approximate_presence_count"),
        "features": g.get("features", []),
        "premium_tier": g.get("premium_tier"),
        "premium_subscription_count": g.get("premium_subscription_count"),
        "verification_level": g.get("verification_level"),
    })


def _list_channels(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all channels in a guild, organized by category."""
    channels = _discord_request("GET", f"/guilds/{guild_id}/channels", token)

    # Organize: categories first, then channels under each
    categories: Dict[Optional[str], Dict[str, Any]] = {}
    uncategorized: List[Dict[str, Any]] = []

    # First pass: collect categories
    for ch in channels:
        if ch["type"] == 4:  # category
            categories[ch["id"]] = {
                "id": ch["id"],
                "name": ch["name"],
                "position": ch.get("position", 0),
                "channels": [],
            }

    # Second pass: assign channels to categories
    for ch in channels:
        if ch["type"] == 4:
            continue
        entry = {
            "id": ch["id"],
            "name": ch.get("name", ""),
            "type": _channel_type_name(ch["type"]),
            "position": ch.get("position", 0),
            "topic": ch.get("topic"),
            "nsfw": ch.get("nsfw", False),
        }
        parent = ch.get("parent_id")
        if parent and parent in categories:
            categories[parent]["channels"].append(entry)
        else:
            uncategorized.append(entry)

    # Sort
    sorted_cats = sorted(categories.values(), key=lambda c: c["position"])
    for cat in sorted_cats:
        cat["channels"].sort(key=lambda c: c["position"])
    uncategorized.sort(key=lambda c: c["position"])

    result: List[Dict[str, Any]] = []
    if uncategorized:
        result.append({"category": None, "channels": uncategorized})
    for cat in sorted_cats:
        result.append({
            "category": {"id": cat["id"], "name": cat["name"]},
            "channels": cat["channels"],
        })

    total = sum(len(group["channels"]) for group in result)
    return json.dumps({"channel_groups": result, "total_channels": total})


def _channel_info(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Get detailed info about a specific channel."""
    ch = _discord_request("GET", f"/channels/{channel_id}", token)
    return json.dumps({
        "id": ch["id"],
        "name": ch.get("name"),
        "type": _channel_type_name(ch["type"]),
        "guild_id": ch.get("guild_id"),
        "topic": ch.get("topic"),
        "nsfw": ch.get("nsfw", False),
        "position": ch.get("position"),
        "parent_id": ch.get("parent_id"),
        "rate_limit_per_user": ch.get("rate_limit_per_user", 0),
        "last_message_id": ch.get("last_message_id"),
    })


def _list_roles(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all roles in a guild."""
    roles = _discord_request("GET", f"/guilds/{guild_id}/roles", token)
    result = []
    for r in sorted(roles, key=lambda r: r.get("position", 0), reverse=True):
        result.append({
            "id": r["id"],
            "name": r["name"],
            "color": f"#{r.get('color', 0):06x}" if r.get("color") else None,
            "position": r.get("position", 0),
            "mentionable": r.get("mentionable", False),
            "managed": r.get("managed", False),
            "member_count": r.get("member_count"),
            "hoist": r.get("hoist", False),
        })
    return json.dumps({"roles": result, "count": len(result)})


def _member_info(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Get info about a specific guild member."""
    m = _discord_request("GET", f"/guilds/{guild_id}/members/{user_id}", token)
    user = m.get("user", {})
    return json.dumps({
        "user_id": user.get("id"),
        "username": user.get("username"),
        "display_name": user.get("global_name"),
        "nickname": m.get("nick"),
        "avatar": user.get("avatar"),
        "bot": user.get("bot", False),
        "roles": m.get("roles", []),
        "joined_at": m.get("joined_at"),
        "premium_since": m.get("premium_since"),
    })


def _search_members(token: str, guild_id: str, query: str, limit: int = 50, **_kwargs: Any) -> str:
    """Search for guild members by name."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20
    params = {"query": query, "limit": str(min(limit, 100))}
    members = _discord_request("GET", f"/guilds/{guild_id}/members/search", token, params=params)
    result = []
    for m in members:
        user = m.get("user", {})
        result.append({
            "user_id": user.get("id"),
            "username": user.get("username"),
            "display_name": user.get("global_name"),
            "nickname": m.get("nick"),
            "bot": user.get("bot", False),
            "roles": m.get("roles", []),
        })
    return json.dumps({"members": result, "count": len(result)})


def _fetch_messages(
    token: str, channel_id: str, limit: int = 50,
    before: Optional[str] = None, after: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Fetch recent messages from a channel."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params: Dict[str, str] = {"limit": str(min(limit, 100))}
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    messages = _discord_request("GET", f"/channels/{channel_id}/messages", token, params=params)
    result = []
    for msg in messages:
        author = msg.get("author", {})
        result.append({
            "id": msg["id"],
            "content": msg.get("content", ""),
            "author": {
                "id": author.get("id"),
                "username": author.get("username"),
                "display_name": author.get("global_name"),
                "bot": author.get("bot", False),
            },
            "timestamp": msg.get("timestamp"),
            "edited_timestamp": msg.get("edited_timestamp"),
            "attachments": [
                {"filename": a.get("filename"), "url": a.get("url"), "size": a.get("size")}
                for a in msg.get("attachments", [])
            ],
            "reactions": [
                {"emoji": r.get("emoji", {}).get("name"), "count": r.get("count", 0)}
                for r in msg.get("reactions", [])
            ] if msg.get("reactions") else [],
            "pinned": msg.get("pinned", False),
        })
    return json.dumps({"messages": result, "count": len(result)})


def _list_pins(token: str, channel_id: str, **_kwargs: Any) -> str:
    """List pinned messages in a channel."""
    messages = _discord_request("GET", f"/channels/{channel_id}/pins", token)
    result = []
    for msg in messages:
        author = msg.get("author", {})
        result.append({
            "id": msg["id"],
            "content": msg.get("content", "")[:200],  # Truncate for overview
            "author": author.get("username"),
            "timestamp": msg.get("timestamp"),
        })
    return json.dumps({"pinned_messages": result, "count": len(result)})


def _pin_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Pin a message in a channel."""
    _discord_request("PUT", f"/channels/{channel_id}/pins/{message_id}", token)
    return json.dumps({"success": True, "message": f"Message {message_id} pinned."})


def _unpin_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Unpin a message from a channel."""
    _discord_request("DELETE", f"/channels/{channel_id}/pins/{message_id}", token)
    return json.dumps({"success": True, "message": f"Message {message_id} unpinned."})


def _delete_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Delete a message from a channel or thread."""
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}", token)
    return json.dumps({"success": True, "message": f"Message {message_id} deleted."})


def _create_thread(
    token: str, channel_id: str, name: str,
    message_id: Optional[str] = None,
    auto_archive_duration: int = 1440,
    **_kwargs: Any,
) -> str:
    """Create a thread in a channel."""
    if message_id:
        # Create thread from an existing message
        path = f"/channels/{channel_id}/messages/{message_id}/threads"
        body: Dict[str, Any] = {
            "name": name,
            "auto_archive_duration": auto_archive_duration,
        }
    else:
        # Create a standalone thread
        path = f"/channels/{channel_id}/threads"
        body = {
            "name": name,
            "auto_archive_duration": auto_archive_duration,
            "type": 11,  # PUBLIC_THREAD
        }
    thread = _discord_request("POST", path, token, body=body)
    return json.dumps({
        "success": True,
        "thread_id": thread["id"],
        "name": thread.get("name"),
    })


def _add_role(token: str, guild_id: str, user_id: str, role_id: str, **_kwargs: Any) -> str:
    """Add a role to a guild member."""
    _discord_request("PUT", f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}", token)
    return json.dumps({"success": True, "message": f"Role {role_id} added to user {user_id}."})


def _remove_role(token: str, guild_id: str, user_id: str, role_id: str, **_kwargs: Any) -> str:
    """Remove a role from a guild member."""
    _discord_request("DELETE", f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}", token)
    return json.dumps({"success": True, "message": f"Role {role_id} removed from user {user_id}."})


# ---------------------------------------------------------------------------
# New: Channel management
# ---------------------------------------------------------------------------

def _create_channel(token: str, guild_id: str, name: str, **_kwargs: Any) -> str:
    """Create a new channel in a guild."""
    body: Dict[str, Any] = {"name": name}
    for key in ("channel_type", "topic", "position", "nsfw", "bitrate", "user_limit",
                "rate_limit_per_user", "parent_id", "permission_overwrites"):
        val = _kwargs.get(key)
        if key == "channel_type" and val is not None:
            body["type"] = int(val)
        elif key == "permission_overwrites" and val is not None:
            try:
                body["permission_overwrites"] = json.loads(val) if isinstance(val, str) else val
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"error": "permission_overwrites must be valid JSON array."})
        elif val is not None:
            body[key] = val
    ch = _discord_request("POST", f"/guilds/{guild_id}/channels", token, body=body)
    return json.dumps({
        "success": True, "id": ch["id"], "name": ch.get("name"), "type": ch.get("type"),
    })


def _modify_channel(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Modify a channel's settings."""
    body: Dict[str, Any] = {}
    for key in ("name", "topic", "position", "nsfw", "bitrate", "user_limit",
                "rate_limit_per_user", "parent_id", "rtc_region", "video_quality_mode",
                "default_auto_archive_duration", "available_tags", "default_reaction_emoji",
                "default_thread_rate_limit_per_user", "default_sort_order", "default_forum_layout",
                "flags"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    # Apply params_json for advanced fields
    raw_params = _kwargs.get("params_json")
    if raw_params:
        try:
            extra = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
            body.update(extra)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "params_json must be valid JSON."})
    if not body:
        return json.dumps({"error": "No channel properties to modify."})
    _discord_request("PATCH", f"/channels/{channel_id}", token, body=body)
    return json.dumps({"success": True, "message": f"Channel {channel_id} modified."})


def _delete_channel(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Delete or close a channel."""
    _discord_request("DELETE", f"/channels/{channel_id}", token)
    return json.dumps({"success": True, "message": f"Channel {channel_id} deleted."})


# ---------------------------------------------------------------------------
# New: Guild management
# ---------------------------------------------------------------------------

def _modify_guild(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Modify a guild's settings."""
    body: Dict[str, Any] = {}
    for key in ("name", "modify_guild_description", "verification_level",
                "default_message_notifications", "explicit_content_filter",
                "afk_channel_id", "afk_timeout", "system_channel_id",
                "system_channel_flags", "preferred_locale", "rules_channel_id",
                "public_updates_channel_id", "premium_progress_bar_enabled",
                "safety_alerts_channel_id"):
        val = _kwargs.get(key)
        if val is not None:
            if key == "modify_guild_description":
                body["description"] = val
            elif key in ("name",):
                body[key] = val
            else:
                body[key] = val
    raw_params = _kwargs.get("params_json")
    if raw_params:
        try:
            extra = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
            body.update(extra)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "params_json must be valid JSON."})
    if not body:
        return json.dumps({"error": "No guild properties to modify."})
    g = _discord_request("PATCH", f"/guilds/{guild_id}", token, body=body)
    return json.dumps({"success": True, "id": g["id"], "name": g.get("name")})


def _get_guild_preview(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get a guild preview for lurkable guilds."""
    g = _discord_request("GET", f"/guilds/{guild_id}/preview", token)
    return json.dumps({
        "id": g["id"], "name": g["name"], "icon": g.get("icon"),
        "features": g.get("features", []),
        "approximate_member_count": g.get("approximate_member_count"),
        "approximate_presence_count": g.get("approximate_presence_count"),
        "stickers": g.get("stickers", []),
        "emojis": g.get("emojis", []),
    })


def _get_guild_vanity_url(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get the guild's vanity URL invite code, if set."""
    result = _discord_request("GET", f"/guilds/{guild_id}/vanity-url", token)
    return json.dumps({"code": result.get("code"), "uses": result.get("uses")})


def _modify_guild_widget(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Modify a guild's widget settings."""
    body: Dict[str, Any] = {}
    for key in ("enabled", "channel_id"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    result = _discord_request("PATCH", f"/guilds/{guild_id}/widget", token, body=body)
    return json.dumps({"success": True, "enabled": result.get("enabled"), "channel_id": result.get("channel_id")})


def _get_guild_widget_settings(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get guild widget settings."""
    result = _discord_request("GET", f"/guilds/{guild_id}/widget-settings", token)
    return json.dumps(result)


def _get_guild_widget(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get guild widget."""
    result = _discord_request("GET", f"/guilds/{guild_id}/widget", token)
    return json.dumps(result)


def _list_guild_members(token: str, guild_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """List all guild members (paginated, requires GUILD_MEMBERS intent)."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params = {"limit": str(min(limit, 1000))}
    after = _kwargs.get("after")
    if after:
        params["after"] = after
    members = _discord_request("GET", f"/guilds/{guild_id}/members", token, params=params)
    result = []
    for m in members:
        user = m.get("user", {})
        result.append({
            "user_id": user.get("id"), "username": user.get("username"),
            "display_name": user.get("global_name"), "nickname": m.get("nick"),
            "bot": user.get("bot", False), "roles": m.get("roles", []),
            "joined_at": m.get("joined_at"),
        })
    return json.dumps({"members": result, "count": len(result)})


def _modify_guild_member(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Modify attributes of a guild member (nick, roles, mute, deaf, channel_id)."""
    body: Dict[str, Any] = {}
    for key in ("nick", "roles", "mute", "deaf", "channel_id", "communication_disabled_until"):
        val = _kwargs.get(key)
        if val is not None:
            if key == "roles" and isinstance(val, str):
                body["roles"] = [r.strip() for r in val.split(",") if r.strip()]
            else:
                body[key] = val
    if not body:
        return json.dumps({"error": "No member properties to modify."})
    _discord_request("PATCH", f"/guilds/{guild_id}/members/{user_id}", token, body=body)
    return json.dumps({"success": True, "message": f"Member {user_id} modified."})


def _kick_guild_member(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Kick a member from the guild."""
    reason = _kwargs.get("reason", "")
    headers = {"X-Audit-Log-Reason": reason} if reason else {}
    _discord_request("DELETE", f"/guilds/{guild_id}/members/{user_id}", token, body=headers)
    return json.dumps({"success": True, "message": f"Member {user_id} kicked."})


# ---------------------------------------------------------------------------
# New: Ban management
# ---------------------------------------------------------------------------

def _ban_user(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Ban a user from the guild."""
    body: Dict[str, Any] = {}
    for key in ("delete_message_days", "delete_message_seconds", "reason"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    _discord_request("PUT", f"/guilds/{guild_id}/bans/{user_id}", token, body=body)
    return json.dumps({"success": True, "message": f"User {user_id} banned."})


def _unban_user(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Remove a ban from a user."""
    _discord_request("DELETE", f"/guilds/{guild_id}/bans/{user_id}", token)
    return json.dumps({"success": True, "message": f"User {user_id} unbanned."})


def _list_guild_bans(token: str, guild_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """List all bans in a guild."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params = {"limit": str(min(limit, 1000))}
    before = _kwargs.get("before")
    after = _kwargs.get("after")
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    bans = _discord_request("GET", f"/guilds/{guild_id}/bans", token, params=params)
    result = []
    for b in bans:
        user = b.get("user", {})
        result.append({
            "user_id": user.get("id"), "username": user.get("username"),
            "reason": b.get("reason"),
        })
    return json.dumps({"bans": result, "count": len(result)})


def _get_guild_ban(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Get a specific ban in a guild."""
    b = _discord_request("GET", f"/guilds/{guild_id}/bans/{user_id}", token)
    user = b.get("user", {})
    return json.dumps({
        "user_id": user.get("id"), "username": user.get("username"),
        "reason": b.get("reason"),
    })


# ---------------------------------------------------------------------------
# New: Role management (CRUD)
# ---------------------------------------------------------------------------

def _create_guild_role(token: str, guild_id: str, name: str, **_kwargs: Any) -> str:
    """Create a new role in the guild."""
    body: Dict[str, Any] = {"name": name}
    for key in ("permissions", "color", "hoist", "mentionable", "icon", "unicode_emoji"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    role = _discord_request("POST", f"/guilds/{guild_id}/roles", token, body=body)
    return json.dumps({"success": True, "id": role["id"], "name": role.get("name")})


def _modify_guild_role(token: str, guild_id: str, role_id: str, **_kwargs: Any) -> str:
    """Modify a role."""
    body: Dict[str, Any] = {}
    for key in ("name", "permissions", "color", "hoist", "mentionable", "icon", "unicode_emoji"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    if not body:
        return json.dumps({"error": "No role properties to modify."})
    role = _discord_request("PATCH", f"/guilds/{guild_id}/roles/{role_id}", token, body=body)
    return json.dumps({"success": True, "id": role["id"], "name": role.get("name")})


def _modify_guild_role_positions(token: str, guild_id: str, role_id: str, position: int, **_kwargs: Any) -> str:
    """Modify the positions of a set of roles. Provide role_id and position."""
    body = [{"id": role_id, "position": position}]
    roles = _discord_request("PATCH", f"/guilds/{guild_id}/roles", token, body=body)
    names = [{"id": r["id"], "name": r["name"], "position": r.get("position")} for r in roles]
    return json.dumps({"success": True, "roles": names})


def _delete_guild_role(token: str, guild_id: str, role_id: str, **_kwargs: Any) -> str:
    """Delete a role."""
    _discord_request("DELETE", f"/guilds/{guild_id}/roles/{role_id}", token)
    return json.dumps({"success": True, "message": f"Role {role_id} deleted."})


# ---------------------------------------------------------------------------
# New: Emoji management
# ---------------------------------------------------------------------------

def _list_guild_emojis(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all emojis in a guild."""
    emojis = _discord_request("GET", f"/guilds/{guild_id}/emojis", token)
    result = []
    for e in emojis:
        result.append({
            "id": e["id"], "name": e["name"], "animated": e.get("animated", False),
            "managed": e.get("managed", False), "available": e.get("available", True),
            "roles": e.get("roles", []),
        })
    return json.dumps({"emojis": result, "count": len(result)})


def _create_guild_emoji(token: str, guild_id: str, name: str, image_data: str, **_kwargs: Any) -> str:
    """Create a new emoji. image_data is base64-encoded image."""
    body: Dict[str, Any] = {"name": name, "image": image_data}
    roles_raw = _kwargs.get("roles")
    if roles_raw:
        body["roles"] = [r.strip() for r in roles_raw.split(",") if r.strip()]
    e = _discord_request("POST", f"/guilds/{guild_id}/emojis", token, body=body)
    return json.dumps({"success": True, "id": e["id"], "name": e.get("name")})


def _modify_guild_emoji(token: str, guild_id: str, emoji_id: str, name: str, **_kwargs: Any) -> str:
    """Modify an emoji's name or roles."""
    body: Dict[str, Any] = {"name": name}
    roles_raw = _kwargs.get("roles")
    if roles_raw is not None:
        body["roles"] = [r.strip() for r in roles_raw.split(",") if r.strip()]
    e = _discord_request("PATCH", f"/guilds/{guild_id}/emojis/{emoji_id}", token, body=body)
    return json.dumps({"success": True, "id": e["id"], "name": e.get("name")})


def _delete_guild_emoji(token: str, guild_id: str, emoji_id: str, **_kwargs: Any) -> str:
    """Delete an emoji from the guild."""
    _discord_request("DELETE", f"/guilds/{guild_id}/emojis/{emoji_id}", token)
    return json.dumps({"success": True, "message": f"Emoji {emoji_id} deleted."})


# ---------------------------------------------------------------------------
# New: Sticker management
# ---------------------------------------------------------------------------

def _list_guild_stickers(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all stickers in a guild."""
    stickers = _discord_request("GET", f"/guilds/{guild_id}/stickers", token)
    result = []
    for s in stickers:
        result.append({
            "id": s["id"], "name": s["name"], "description": s.get("description"),
            "tags": s.get("tags"), "type": s.get("type"), "available": s.get("available", True),
        })
    return json.dumps({"stickers": result, "count": len(result)})


def _create_guild_sticker(token: str, guild_id: str, name: str, sticker_tags: str, image_data: str, **_kwargs: Any) -> str:
    """Create a new sticker. image_data is base64-encoded PNG/APNG/Lottie. sticker_tags = autocomplete tags."""
    body: Dict[str, Any] = {
        "name": name, "tags": sticker_tags, "image": image_data,
    }
    desc = _kwargs.get("sticker_description")
    if desc:
        body["description"] = desc
    s = _discord_request("POST", f"/guilds/{guild_id}/stickers", token, body=body)
    return json.dumps({"success": True, "id": s["id"], "name": s.get("name")})


def _modify_guild_sticker(token: str, guild_id: str, sticker_id: str, **_kwargs: Any) -> str:
    """Modify a sticker's name, description, or tags."""
    body: Dict[str, Any] = {}
    for key, api_key in (("name", "name"), ("sticker_description", "description"), ("sticker_tags", "tags")):
        val = _kwargs.get(key)
        if val is not None:
            body[api_key] = val
    if not body:
        return json.dumps({"error": "No sticker properties to modify."})
    s = _discord_request("PATCH", f"/guilds/{guild_id}/stickers/{sticker_id}", token, body=body)
    return json.dumps({"success": True, "id": s["id"], "name": s.get("name")})


def _delete_guild_sticker(token: str, guild_id: str, sticker_id: str, **_kwargs: Any) -> str:
    """Delete a sticker from the guild."""
    _discord_request("DELETE", f"/guilds/{guild_id}/stickers/{sticker_id}", token)
    return json.dumps({"success": True, "message": f"Sticker {sticker_id} deleted."})


# ---------------------------------------------------------------------------
# New: Webhook management
# ---------------------------------------------------------------------------

def _list_channel_webhooks(token: str, channel_id: str, **_kwargs: Any) -> str:
    """List webhooks for a channel."""
    webhooks = _discord_request("GET", f"/channels/{channel_id}/webhooks", token)
    result = []
    for w in webhooks:
        result.append({
            "id": w["id"], "name": w.get("name"), "channel_id": w.get("channel_id"),
            "guild_id": w.get("guild_id"), "type": w.get("type"),
            "avatar": w.get("avatar"),
        })
    return json.dumps({"webhooks": result, "count": len(result)})


def _list_guild_webhooks(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List webhooks for a guild."""
    webhooks = _discord_request("GET", f"/guilds/{guild_id}/webhooks", token)
    result = []
    for w in webhooks:
        result.append({
            "id": w["id"], "name": w.get("name"), "channel_id": w.get("channel_id"),
            "guild_id": w.get("guild_id"), "type": w.get("type"),
            "avatar": w.get("avatar"),
        })
    return json.dumps({"webhooks": result, "count": len(result)})


def _create_webhook(token: str, channel_id: str, name: str, **_kwargs: Any) -> str:
    """Create a webhook in a channel."""
    body: Dict[str, Any] = {"name": name}
    avatar = _kwargs.get("webhook_avatar")
    if avatar:
        body["avatar"] = avatar
    w = _discord_request("POST", f"/channels/{channel_id}/webhooks", token, body=body)
    return json.dumps({"success": True, "id": w["id"], "name": w.get("name")})


def _get_webhook(token: str, webhook_id: str, **_kwargs: Any) -> str:
    """Get a webhook by ID."""
    w = _discord_request("GET", f"/webhooks/{webhook_id}", token)
    return json.dumps({
        "id": w["id"], "name": w.get("name"), "channel_id": w.get("channel_id"),
        "guild_id": w.get("guild_id"), "type": w.get("type"), "avatar": w.get("avatar"),
    })


def _modify_webhook(token: str, webhook_id: str, **_kwargs: Any) -> str:
    """Modify a webhook."""
    body: Dict[str, Any] = {}
    for key in ("name", "channel_id"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    avatar = _kwargs.get("webhook_avatar")
    if avatar:
        body["avatar"] = avatar
    if not body:
        return json.dumps({"error": "No webhook properties to modify."})
    w = _discord_request("PATCH", f"/webhooks/{webhook_id}", token, body=body)
    return json.dumps({"success": True, "id": w["id"], "name": w.get("name")})


def _delete_webhook(token: str, webhook_id: str, **_kwargs: Any) -> str:
    """Delete a webhook permanently."""
    _discord_request("DELETE", f"/webhooks/{webhook_id}", token)
    return json.dumps({"success": True, "message": f"Webhook {webhook_id} deleted."})


# ---------------------------------------------------------------------------
# New: Thread management
# ---------------------------------------------------------------------------

def _list_active_threads(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List active threads in a guild."""
    threads = _discord_request("GET", f"/guilds/{guild_id}/threads/active", token)
    result = []
    for t in threads.get("threads", []):
        result.append({
            "id": t["id"], "name": t.get("name"), "type": t.get("type"),
            "thread_metadata": {
                "archived": t.get("thread_metadata", {}).get("archived"),
                "auto_archive_duration": t.get("thread_metadata", {}).get("auto_archive_duration"),
                "archive_timestamp": t.get("thread_metadata", {}).get("archive_timestamp"),
            },
            "member_count": t.get("member_count"),
            "message_count": t.get("message_count"),
        })
    return json.dumps({"active_threads": result, "count": len(result)})


def _list_public_archived_threads(token: str, channel_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """List public archived threads in a channel."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params = {"limit": str(min(limit, 100))}
    before = _kwargs.get("before")
    if before:
        params["before"] = before
    threads = _discord_request("GET", f"/channels/{channel_id}/threads/archived/public", token, params=params)
    result = []
    for t in threads.get("threads", []):
        result.append({
            "id": t["id"], "name": t.get("name"),
            "archived": t.get("thread_metadata", {}).get("archived"),
            "auto_archive_duration": t.get("thread_metadata", {}).get("auto_archive_duration"),
            "archive_timestamp": t.get("thread_metadata", {}).get("archive_timestamp"),
        })
    return json.dumps({"threads": result, "count": len(result), "has_more": threads.get("has_more", False)})


def _list_private_archived_threads(token: str, channel_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """List private archived threads in a channel."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params = {"limit": str(min(limit, 100))}
    before = _kwargs.get("before")
    if before:
        params["before"] = before
    threads = _discord_request("GET", f"/channels/{channel_id}/threads/archived/private", token, params=params)
    result = []
    for t in threads.get("threads", []):
        result.append({
            "id": t["id"], "name": t.get("name"),
            "archived": t.get("thread_metadata", {}).get("archived"),
            "auto_archive_duration": t.get("thread_metadata", {}).get("auto_archive_duration"),
            "archive_timestamp": t.get("thread_metadata", {}).get("archive_timestamp"),
        })
    return json.dumps({"threads": result, "count": len(result), "has_more": threads.get("has_more", False)})


def _list_thread_members(token: str, channel_id: str, **_kwargs: Any) -> str:
    """List members of a thread."""
    members = _discord_request("GET", f"/channels/{channel_id}/thread-members", token)
    result = []
    for m in members:
        user = m.get("user", {})
        result.append({
            "user_id": user.get("id"), "username": user.get("username"),
            "join_timestamp": m.get("join_timestamp"),
            "flags": m.get("flags", 0),
        }) if user else result.append({
            "user_id": m.get("id"), "user": m,
        })
    return json.dumps({"members": result, "count": len(result)})


def _manage_thread_member(token: str, channel_id: str, user_id: str, action: str, **_kwargs: Any) -> str:
    """Add or remove a member from a thread. action='add' or 'remove'."""
    if action == "add":
        _discord_request("PUT", f"/channels/{channel_id}/thread-members/{user_id}", token)
        return json.dumps({"success": True, "message": f"User {user_id} added to thread."})
    elif action == "remove":
        _discord_request("DELETE", f"/channels/{channel_id}/thread-members/{user_id}", token)
        return json.dumps({"success": True, "message": f"User {user_id} removed from thread."})
    return json.dumps({"error": "action must be 'add' or 'remove' for manage_thread_member."})


def _modify_thread(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Modify a thread's settings (archived, auto_archive_duration, locked, etc.)."""
    body: Dict[str, Any] = {}
    for key in ("name", "archived", "auto_archive_duration", "locked", "invitable",
                "rate_limit_per_user", "flags", "applied_tags"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    if not body:
        return json.dumps({"error": "No thread properties to modify."})
    _discord_request("PATCH", f"/channels/{channel_id}", token, body=body)
    return json.dumps({"success": True, "message": f"Thread {channel_id} modified."})


# ---------------------------------------------------------------------------
# New: Message moderation
# ---------------------------------------------------------------------------

def _bulk_delete_messages(token: str, channel_id: str, message_ids: str, **_kwargs: Any) -> str:
    """Bulk delete messages in a channel. message_ids = comma-separated IDs (2-100)."""
    ids = [m.strip() for m in message_ids.split(",") if m.strip()]
    if len(ids) < 2 or len(ids) > 100:
        return json.dumps({"error": "Must provide between 2 and 100 message IDs."})
    _discord_request("POST", f"/channels/{channel_id}/messages/bulk-delete", token, body={"messages": ids})
    return json.dumps({"success": True, "message": f"{len(ids)} messages deleted."})


def _crosspost_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Crosspost (publish) a message in an announcement channel."""
    result = _discord_request("POST", f"/channels/{channel_id}/messages/{message_id}/crosspost", token)
    return json.dumps({"success": True, "id": result.get("id"), "timestamp": result.get("timestamp")})


# ---------------------------------------------------------------------------
# New: Auto-moderation
# ---------------------------------------------------------------------------

def _list_auto_mod_rules(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List auto-moderation rules for a guild."""
    rules = _discord_request("GET", f"/guilds/{guild_id}/auto-moderation/rules", token)
    result = []
    for r in rules:
        result.append({
            "id": r["id"], "name": r.get("name"), "event_type": r.get("event_type"),
            "trigger_type": r.get("trigger_type"), "enabled": r.get("enabled", False),
            "actions": [
                {"type": a.get("type"), "metadata": a.get("metadata")}
                for a in r.get("actions", [])
            ],
        })
    return json.dumps({"rules": result, "count": len(result)})


def _create_auto_mod_rule(token: str, guild_id: str, name: str, event_type: int, trigger_type: int, actions_json: str, **_kwargs: Any) -> str:
    """Create an auto-moderation rule. actions_json = JSON array of action objects."""
    try:
        actions = json.loads(actions_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "actions_json must be valid JSON array."})
    body: Dict[str, Any] = {
        "name": name, "event_type": event_type, "trigger_type": trigger_type,
        "actions": actions,
    }
    trigger_metadata = _kwargs.get("trigger_metadata")
    if trigger_metadata:
        try:
            body["trigger_metadata"] = json.loads(trigger_metadata)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "trigger_metadata must be valid JSON."})
    keyword_filter = _kwargs.get("keyword_filter")
    if keyword_filter:
        body.setdefault("trigger_metadata", {})["keyword_filter"] = [
            k.strip() for k in keyword_filter.split(",")
        ]
    enabled = _kwargs.get("enabled")
    if enabled is not None:
        body["enabled"] = enabled
    exempt_roles = _kwargs.get("exempt_roles")
    if exempt_roles:
        body["exempt_roles"] = [r.strip() for r in exempt_roles.split(",") if r.strip()]
    exempt_channels = _kwargs.get("exempt_channels")
    if exempt_channels:
        body["exempt_channels"] = [c.strip() for c in exempt_channels.split(",") if c.strip()]
    r = _discord_request("POST", f"/guilds/{guild_id}/auto-moderation/rules", token, body=body)
    return json.dumps({"success": True, "id": r["id"], "name": r.get("name")})


def _modify_auto_mod_rule(token: str, guild_id: str, rule_id: str, **_kwargs: Any) -> str:
    """Modify an auto-moderation rule."""
    body: Dict[str, Any] = {}
    for key in ("name", "event_type", "trigger_type", "enabled"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    actions_raw = _kwargs.get("actions_json")
    if actions_raw:
        try:
            body["actions"] = json.loads(actions_raw)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "actions_json must be valid JSON."})
    keyword_filter = _kwargs.get("keyword_filter")
    if keyword_filter:
        body.setdefault("trigger_metadata", {})["keyword_filter"] = [
            k.strip() for k in keyword_filter.split(",")
        ]
    exempt_roles = _kwargs.get("exempt_roles")
    if exempt_roles:
        body["exempt_roles"] = [r.strip() for r in exempt_roles.split(",") if r.strip()]
    exempt_channels = _kwargs.get("exempt_channels")
    if exempt_channels:
        body["exempt_channels"] = [c.strip() for c in exempt_channels.split(",") if c.strip()]
    if not body:
        return json.dumps({"error": "No rule properties to modify."})
    r = _discord_request("PATCH", f"/guilds/{guild_id}/auto-moderation/rules/{rule_id}", token, body=body)
    return json.dumps({"success": True, "id": r["id"], "name": r.get("name")})


def _delete_auto_mod_rule(token: str, guild_id: str, rule_id: str, **_kwargs: Any) -> str:
    """Delete an auto-moderation rule."""
    _discord_request("DELETE", f"/guilds/{guild_id}/auto-moderation/rules/{rule_id}", token)
    return json.dumps({"success": True, "message": f"Auto-mod rule {rule_id} deleted."})


# ---------------------------------------------------------------------------
# New: Scheduled events
# ---------------------------------------------------------------------------

def _list_scheduled_events(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List scheduled events for a guild."""
    params = {}
    with_user_count = _kwargs.get("with_user_count")
    if with_user_count:
        params["with_user_count"] = "true"
    events = _discord_request("GET", f"/guilds/{guild_id}/scheduled-events", token, params=params)
    result = []
    for e in events:
        result.append({
            "id": e["id"], "name": e.get("name"), "description": e.get("description"),
            "scheduled_start_time": e.get("scheduled_start_time"),
            "scheduled_end_time": e.get("scheduled_end_time"),
            "entity_type": e.get("entity_type"), "entity_id": e.get("entity_id"),
            "status": e.get("status"), "creator_id": e.get("creator_id"),
            "user_count": e.get("user_count"),
        })
    return json.dumps({"events": result, "count": len(result)})


def _create_scheduled_event(token: str, guild_id: str, name: str, scheduled_start_time: str, entity_type: int, **_kwargs: Any) -> str:
    """Create a scheduled event. entity_type: 1=stage, 2=voice, 3=external."""
    body: Dict[str, Any] = {
        "name": name, "scheduled_start_time": scheduled_start_time, "entity_type": entity_type,
    }
    for key in ("event_description", "scheduled_end_time", "channel_id",
                "entity_metadata", "privacy_level"):
        val = _kwargs.get(key)
        if val is not None:
            api_key = "description" if key == "event_description" else key
            if key == "entity_metadata" and isinstance(val, str):
                try:
                    body["entity_metadata"] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return json.dumps({"error": "entity_metadata must be valid JSON."})
            else:
                body[api_key] = val
    e = _discord_request("POST", f"/guilds/{guild_id}/scheduled-events", token, body=body)
    return json.dumps({"success": True, "id": e["id"], "name": e.get("name")})


def _modify_scheduled_event(token: str, guild_id: str, event_id: str, **_kwargs: Any) -> str:
    """Modify a scheduled event."""
    body: Dict[str, Any] = {}
    for key, api_key in (
        ("name", "name"), ("event_description", "description"), ("scheduled_start_time", "scheduled_start_time"),
        ("scheduled_end_time", "scheduled_end_time"), ("entity_type", "entity_type"),
        ("status", "status"), ("channel_id", "channel_id"), ("privacy_level", "privacy_level"),
    ):
        val = _kwargs.get(key)
        if val is not None:
            body[api_key] = val
    entity_metadata = _kwargs.get("entity_metadata")
    if entity_metadata:
        try:
            body["entity_metadata"] = json.loads(entity_metadata)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "entity_metadata must be valid JSON."})
    if not body:
        return json.dumps({"error": "No event properties to modify."})
    e = _discord_request("PATCH", f"/guilds/{guild_id}/scheduled-events/{event_id}", token, body=body)
    return json.dumps({"success": True, "id": e["id"], "name": e.get("name")})


def _delete_scheduled_event(token: str, guild_id: str, event_id: str, **_kwargs: Any) -> str:
    """Delete a scheduled event."""
    _discord_request("DELETE", f"/guilds/{guild_id}/scheduled-events/{event_id}", token)
    return json.dumps({"success": True, "message": f"Event {event_id} deleted."})


# ---------------------------------------------------------------------------
# New: Invite management
# ---------------------------------------------------------------------------

def _create_channel_invite(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Create an invite for a channel."""
    body: Dict[str, Any] = {}
    for key in ("invite_max_age", "invite_max_uses", "invite_temporary",
                "invite_unique", "target_type", "target_user_id",
                "target_application_id"):
        val = _kwargs.get(key)
        if val is not None:
            api_key = key.replace("invite_", "", 1) if key.startswith("invite_") else key
            body[api_key] = val
    invite = _discord_request("POST", f"/channels/{channel_id}/invites", token, body=body)
    return json.dumps({
        "success": True, "code": invite.get("code"), "max_age": invite.get("max_age"),
        "max_uses": invite.get("max_uses"), "temporary": invite.get("temporary", False),
    })


def _delete_invite(token: str, invite_code: str, **_kwargs: Any) -> str:
    """Delete an invite by code."""
    _discord_request("DELETE", f"/invites/{invite_code}", token)
    return json.dumps({"success": True, "message": f"Invite {invite_code} deleted."})


# ---------------------------------------------------------------------------
# New: Guild templates
# ---------------------------------------------------------------------------

def _list_guild_templates(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all templates for a guild."""
    templates = _discord_request("GET", f"/guilds/{guild_id}/templates", token)
    result = []
    for t in templates:
        result.append({
            "code": t["code"], "name": t.get("name"), "description": t.get("description"),
            "usage_count": t.get("usage_count"), "creator_id": t.get("creator_id"),
            "created_at": t.get("created_at"), "updated_at": t.get("updated_at"),
        })
    return json.dumps({"templates": result, "count": len(result)})


def _create_guild_template(token: str, guild_id: str, name: str, **_kwargs: Any) -> str:
    """Create a guild template."""
    body: Dict[str, Any] = {"name": name}
    desc = _kwargs.get("template_description")
    if desc:
        body["description"] = desc
    t = _discord_request("POST", f"/guilds/{guild_id}/templates", token, body=body)
    return json.dumps({"success": True, "code": t["code"], "name": t.get("name")})


def _modify_guild_template(token: str, guild_id: str, template_code: str, **_kwargs: Any) -> str:
    """Modify a guild template."""
    body: Dict[str, Any] = {}
    name = _kwargs.get("name")
    if name:
        body["name"] = name
    desc = _kwargs.get("template_description")
    if desc:
        body["description"] = desc
    if not body:
        return json.dumps({"error": "No template properties to modify."})
    t = _discord_request("PATCH", f"/guilds/{guild_id}/templates/{template_code}", token, body=body)
    return json.dumps({"success": True, "code": t["code"], "name": t.get("name")})


def _delete_guild_template(token: str, guild_id: str, template_code: str, **_kwargs: Any) -> str:
    """Delete a guild template."""
    _discord_request("DELETE", f"/guilds/{guild_id}/templates/{template_code}", token)
    return json.dumps({"success": True, "message": f"Template {template_code} deleted."})


def _sync_guild_template(token: str, guild_id: str, template_code: str, **_kwargs: Any) -> str:
    """Sync a guild template to the current guild state."""
    t = _discord_request("PUT", f"/guilds/{guild_id}/templates/{template_code}", token)
    return json.dumps({"success": True, "code": t["code"], "name": t.get("name")})


def _leave_guild(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Leave a guild."""
    _discord_request("DELETE", f"/users/@me/guilds/{guild_id}", token)
    return json.dumps({"success": True, "message": f"Left guild {guild_id}."})


# ---------------------------------------------------------------------------
# More guild: audit log, integrations, prune, onboarding, voice, welcome screen
# ---------------------------------------------------------------------------

def _get_guild_audit_log(token: str, guild_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """Get audit log entries for a guild."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params = {"limit": str(min(limit, 100))}
    for key in ("user_id", "action_type", "before", "after"):
        val = _kwargs.get(key)
        if val:
            params[key] = val
    entries = _discord_request("GET", f"/guilds/{guild_id}/audit-logs", token, params=params)
    result = []
    for e in entries.get("audit_log_entries", []):
        result.append({
            "id": e["id"], "action_type": e.get("action_type"),
            "user_id": e.get("user_id"), "target_id": e.get("target_id"),
            "reason": e.get("reason"),
            "changes": e.get("changes", [])[:3],  # limit context
        })
    return json.dumps({"audit_log_entries": result, "count": len(result)})


def _get_guild_welcome_screen(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get the guild's welcome screen."""
    result = _discord_request("GET", f"/guilds/{guild_id}/welcome-screen", token)
    return json.dumps(result)


def _modify_guild_welcome_screen(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Modify the guild's welcome screen."""
    body: Dict[str, Any] = {}
    enabled = _kwargs.get("enabled")
    if enabled is not None:
        body["enabled"] = enabled
    desc = _kwargs.get("modify_guild_description")
    if desc:
        body["description"] = desc
    raw = _kwargs.get("params_json")
    if raw:
        try:
            extra = json.loads(raw) if isinstance(raw, str) else raw
            body.update(extra)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "params_json must be valid JSON."})
    result = _discord_request("PATCH", f"/guilds/{guild_id}/welcome-screen", token, body=body)
    return json.dumps(result)


def _get_guild_onboarding(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get the guild's onboarding configuration."""
    result = _discord_request("GET", f"/guilds/{guild_id}/onboarding", token)
    return json.dumps(result)


def _modify_guild_onboarding(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Modify the guild's onboarding configuration."""
    body: Dict[str, Any] = {}
    for key in ("prompts", "default_channel_ids", "enabled", "mode"):
        val = _kwargs.get(key)
        if val is not None:
            if isinstance(val, str) and key in ("prompts", "default_channel_ids"):
                try:
                    body[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return json.dumps({"error": f"{key} must be valid JSON."})
            else:
                body[key] = val
    if not body:
        return json.dumps({"error": "No onboarding properties to modify."})
    result = _discord_request("PUT", f"/guilds/{guild_id}/onboarding", token, body=body)
    return json.dumps(result)


def _get_guild_voice_regions(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get voice regions for a guild."""
    regions = _discord_request("GET", f"/guilds/{guild_id}/regions", token)
    return json.dumps({"regions": regions})


def _get_guild_integrations(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get integrations for a guild."""
    integrations = _discord_request("GET", f"/guilds/{guild_id}/integrations", token)
    result = []
    for i in integrations:
        result.append({
            "id": i["id"], "type": i.get("type"), "name": i.get("name"),
            "enabled": i.get("enabled", False), "syncing": i.get("syncing", False),
            "role_id": i.get("role_id"),
        })
    return json.dumps({"integrations": result, "count": len(result)})


def _delete_guild_integration(token: str, guild_id: str, integration_id: str, **_kwargs: Any) -> str:
    """Delete a guild integration."""
    _discord_request("DELETE", f"/guilds/{guild_id}/integrations/{integration_id}", token)
    return json.dumps({"success": True, "message": f"Integration {integration_id} deleted."})


def _sync_guild_integration(token: str, guild_id: str, integration_id: str, **_kwargs: Any) -> str:
    """Sync a guild integration."""
    _discord_request("POST", f"/guilds/{guild_id}/integrations/{integration_id}/sync", token)
    return json.dumps({"success": True, "message": f"Integration {integration_id} synced."})


def _get_guild_prune_count(token: str, guild_id: str, days: int = 7, **_kwargs: Any) -> str:
    """Get count of members that would be pruned."""
    params = {"days": str(days)}
    include_roles = _kwargs.get("include_roles")
    if include_roles:
        params["include_roles"] = include_roles
    result = _discord_request("GET", f"/guilds/{guild_id}/prune", token, params=params)
    return json.dumps({"pruned": result.get("pruned", 0)})


def _begin_guild_prune(token: str, guild_id: str, days: int = 7, **_kwargs: Any) -> str:
    """Begin pruning inactive members."""
    body: Dict[str, Any] = {"days": days}
    compute = _kwargs.get("compute_prune_count")
    if compute is not None:
        body["compute_prune_count"] = compute
    reason = _kwargs.get("reason")
    if reason:
        body["reason"] = reason
    result = _discord_request("POST", f"/guilds/{guild_id}/prune", token, body=body)
    return json.dumps({"pruned": result.get("pruned", 0)})


def _modify_guild_mfa_level(token: str, guild_id: str, level: int, **_kwargs: Any) -> str:
    """Modify the MFA level requirement for the guild."""
    result = _discord_request("POST", f"/guilds/{guild_id}/mfa", token, body={"level": level})
    return json.dumps({"level": result.get("level")})


def _modify_user_voice_state(token: str, guild_id: str, user_id: str, channel_id: str, **_kwargs: Any) -> str:
    """Modify a user's voice state (move, suppress)."""
    body: Dict[str, Any] = {"channel_id": channel_id}
    suppress = _kwargs.get("suppress")
    if suppress is not None:
        body["suppress"] = suppress
    _discord_request("PATCH", f"/guilds/{guild_id}/voice-states/{user_id}", token, body=body)
    return json.dumps({"success": True, "message": f"User {user_id} voice state modified."})


def _modify_current_user_voice_state(token: str, guild_id: str, channel_id: str, **_kwargs: Any) -> str:
    """Modify the bot's own voice state."""
    body: Dict[str, Any] = {"channel_id": channel_id}
    suppress = _kwargs.get("suppress")
    if suppress is not None:
        body["suppress"] = suppress
    _discord_request("PATCH", f"/guilds/{guild_id}/voice-states/@me", token, body=body)
    return json.dumps({"success": True, "message": "Bot voice state modified."})


def _delete_guild(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Delete a guild permanently (owner only)."""
    _discord_request("DELETE", f"/guilds/{guild_id}", token)
    return json.dumps({"success": True, "message": f"Guild {guild_id} deleted."})


def _create_guild(token: str, name: str, **_kwargs: Any) -> str:
    """Create a new guild. Bot must be in fewer than 10 guilds."""
    body: Dict[str, Any] = {"name": name}
    for key in ("icon", "verification_level", "default_message_notifications", "explicit_content_filter"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    # optional roles / channels via params_json since they're complex
    raw = _kwargs.get("params_json")
    if raw:
        try:
            extra = json.loads(raw) if isinstance(raw, str) else raw
            body.update(extra)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "params_json must be valid JSON."})
    g = _discord_request("POST", "/guilds", token, body=body)
    return json.dumps({"success": True, "id": g["id"], "name": g.get("name")})


# ---------------------------------------------------------------------------
# More channel: permissions, positions, invites, reactions, messages
# ---------------------------------------------------------------------------

def _modify_guild_channel_positions(token: str, guild_id: str, channel_positions: str, **_kwargs: Any) -> str:
    """Modify channel positions in a guild. channel_positions = JSON array of {id, position} objects."""
    try:
        positions = json.loads(channel_positions) if isinstance(channel_positions, str) else channel_positions
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "channel_positions must be valid JSON array."})
    _discord_request("PATCH", f"/guilds/{guild_id}/channels", token, body=positions)
    return json.dumps({"success": True, "message": "Channel positions updated."})


def _edit_channel_permissions(token: str, channel_id: str, overwrite_id: str, allow: str, deny: str, **_kwargs: Any) -> str:
    """Edit permission overwrites for a role or member in a channel."""
    ptype = _kwargs.get("ptype", 0)
    body = {"allow": allow, "deny": deny, "type": int(ptype) if ptype else 0}
    _discord_request("PUT", f"/channels/{channel_id}/permissions/{overwrite_id}", token, body=body)
    return json.dumps({"success": True, "message": f"Permission overwrite {overwrite_id} updated."})


def _delete_channel_permission(token: str, channel_id: str, overwrite_id: str, **_kwargs: Any) -> str:
    """Delete a permission overwrite from a channel."""
    _discord_request("DELETE", f"/channels/{channel_id}/permissions/{overwrite_id}", token)
    return json.dumps({"success": True, "message": f"Permission overwrite {overwrite_id} deleted."})


def _get_channel_invites(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Get all active invites for a channel."""
    invites = _discord_request("GET", f"/channels/{channel_id}/invites", token)
    result = []
    for i in invites:
        result.append({
            "code": i.get("code"), "max_age": i.get("max_age"),
            "max_uses": i.get("max_uses"), "uses": i.get("uses", 0),
            "temporary": i.get("temporary", False),
            "created_at": i.get("created_at"),
            "inviter": {"id": i.get("inviter", {}).get("id"), "username": i.get("inviter", {}).get("username")},
        })
    return json.dumps({"invites": result, "count": len(result)})


def _follow_announcement_channel(token: str, channel_id: str, webhook_channel_id: str, **_kwargs: Any) -> str:
    """Follow an announcement channel to send messages to another channel."""
    result = _discord_request("POST", f"/channels/{channel_id}/followers", token, body={"webhook_channel_id": webhook_channel_id})
    return json.dumps({"success": True, "channel_id": result.get("channel_id")})


def _get_invite(token: str, invite_code: str, **_kwargs: Any) -> str:
    """Get details about an invite by its code."""
    params = {}
    with_counts = _kwargs.get("with_counts")
    if with_counts:
        params["with_counts"] = "true"
    i = _discord_request("GET", f"/invites/{invite_code}", token, params=params)
    return json.dumps({
        "code": i.get("code"), "guild_id": i.get("guild", {}).get("id"),
        "channel_id": i.get("channel", {}).get("id"),
        "approximate_member_count": i.get("approximate_member_count"),
        "approximate_presence_count": i.get("approximate_presence_count"),
        "expires_at": i.get("expires_at"),
    })


# ---------------------------------------------------------------------------
# Message actions: get single message, edit message, reactions
# ---------------------------------------------------------------------------

def _get_channel_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Get a specific message by ID."""
    msg = _discord_request("GET", f"/channels/{channel_id}/messages/{message_id}", token)
    author = msg.get("author", {})
    return json.dumps({
        "id": msg["id"], "content": msg.get("content", ""),
        "author": {"id": author.get("id"), "username": author.get("username"), "bot": author.get("bot", False)},
        "timestamp": msg.get("timestamp"), "pinned": msg.get("pinned", False),
        "attachments": [{"filename": a.get("filename"), "url": a.get("url")} for a in msg.get("attachments", [])],
        "mentions": [{"id": m.get("id"), "username": m.get("username")} for m in msg.get("mentions", [])],
        "reactions": [{"emoji": r.get("emoji", {}).get("name"), "count": r.get("count", 0)} for r in msg.get("reactions", [])],
    })


def _edit_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Edit a previously sent message."""
    body: Dict[str, Any] = {}
    for key in ("content", "flags"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    raw = _kwargs.get("params_json")
    if raw:
        try:
            extra = json.loads(raw) if isinstance(raw, str) else raw
            body.update(extra)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "params_json must be valid JSON."})
    if not body:
        return json.dumps({"error": "No message properties to modify."})
    msg = _discord_request("PATCH", f"/channels/{channel_id}/messages/{message_id}", token, body=body)
    return json.dumps({"success": True, "id": msg.get("id")})


def _add_reaction(token: str, channel_id: str, message_id: str, emoji: str, **_kwargs: Any) -> str:
    """Add a reaction to a message. emoji = URL-encoded emoji name or unicode."""
    _discord_request("PUT", f"/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/@me", token)
    return json.dumps({"success": True, "message": f"Reaction {emoji} added."})


def _remove_own_reaction(token: str, channel_id: str, message_id: str, emoji: str, **_kwargs: Any) -> str:
    """Remove own reaction from a message."""
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/@me", token)
    return json.dumps({"success": True, "message": f"Reaction {emoji} removed."})


def _remove_user_reaction(token: str, channel_id: str, message_id: str, emoji: str, user_id: str, **_kwargs: Any) -> str:
    """Remove another user's reaction from a message."""
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/{user_id}", token)
    return json.dumps({"success": True, "message": f"Reaction {emoji} removed for user {user_id}."})


def _get_reactions(token: str, channel_id: str, message_id: str, emoji: str, limit: int = 25, **_kwargs: Any) -> str:
    """Get users who reacted with a specific emoji."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 25
    params = {"limit": str(min(limit, 100))}
    after = _kwargs.get("after")
    if after:
        params["after"] = after
    users = _discord_request("GET", f"/channels/{channel_id}/messages/{message_id}/reactions/{emoji}", token, params=params)
    result = [{"id": u["id"], "username": u.get("username")} for u in users]
    return json.dumps({"users": result, "count": len(result)})


def _delete_all_reactions(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Remove all reactions from a message."""
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}/reactions", token)
    return json.dumps({"success": True, "message": "All reactions removed."})


def _delete_all_reactions_for_emoji(token: str, channel_id: str, message_id: str, emoji: str, **_kwargs: Any) -> str:
    """Remove all reactions for a specific emoji from a message."""
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}/reactions/{emoji}", token)
    return json.dumps({"success": True, "message": f"All reactions for {emoji} removed."})


# ---------------------------------------------------------------------------
# Thread: join, leave, get member, list joined archived
# ---------------------------------------------------------------------------

def _join_thread(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Add the bot to a thread."""
    _discord_request("PUT", f"/channels/{channel_id}/thread-members/@me", token)
    return json.dumps({"success": True, "message": "Joined thread."})


def _leave_thread(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Remove the bot from a thread."""
    _discord_request("DELETE", f"/channels/{channel_id}/thread-members/@me", token)
    return json.dumps({"success": True, "message": "Left thread."})


def _get_thread_member(token: str, channel_id: str, user_id: str, **_kwargs: Any) -> str:
    """Get a specific thread member."""
    m = _discord_request("GET", f"/channels/{channel_id}/thread-members/{user_id}", token)
    user = m.get("user", {})
    return json.dumps({
        "user_id": user.get("id"), "username": user.get("username"),
        "join_timestamp": m.get("join_timestamp"), "flags": m.get("flags", 0),
    })


def _list_joined_private_archived_threads(token: str, channel_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """List joined private archived threads."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params = {"limit": str(min(limit, 100))}
    before = _kwargs.get("before")
    if before:
        params["before"] = before
    threads = _discord_request("GET", f"/channels/{channel_id}/users/@me/threads/archived/private", token, params=params)
    result = []
    for t in threads.get("threads", []):
        result.append({
            "id": t["id"], "name": t.get("name"),
            "archived": t.get("thread_metadata", {}).get("archived"),
            "auto_archive_duration": t.get("thread_metadata", {}).get("auto_archive_duration"),
        })
    return json.dumps({"threads": result, "count": len(result), "has_more": threads.get("has_more", False)})


# ---------------------------------------------------------------------------
# Member: modify own nickname, add via OAuth2
# ---------------------------------------------------------------------------

def _modify_current_member(token: str, guild_id: str, nick: str, **_kwargs: Any) -> str:
    """Modify the bot's own nickname in the guild."""
    _discord_request("PATCH", f"/guilds/{guild_id}/members/@me", token, body={"nick": nick})
    return json.dumps({"success": True, "message": f"Nickname changed to '{nick}'."})


def _add_guild_member(token: str, guild_id: str, user_id: str, access_token: str, **_kwargs: Any) -> str:
    """Add a user to the guild using an OAuth2 access token."""
    body: Dict[str, Any] = {"access_token": access_token}
    for key in ("nick", "roles", "mute", "deaf"):
        val = _kwargs.get(key)
        if val is not None:
            body[key] = val
    result = _discord_request("PUT", f"/guilds/{guild_id}/members/{user_id}", token, body=body)
    return json.dumps({"success": True, "id": result.get("user", {}).get("id") if isinstance(result, dict) else None})


# ---------------------------------------------------------------------------
# Single-item reads: emoji, sticker, auto-mod rule, scheduled event, template
# ---------------------------------------------------------------------------

def _get_guild_emoji(token: str, guild_id: str, emoji_id: str, **_kwargs: Any) -> str:
    """Get a specific guild emoji."""
    e = _discord_request("GET", f"/guilds/{guild_id}/emojis/{emoji_id}", token)
    return json.dumps({
        "id": e["id"], "name": e["name"], "animated": e.get("animated", False),
        "managed": e.get("managed", False), "available": e.get("available", True),
        "roles": e.get("roles", []),
    })


def _get_guild_sticker(token: str, guild_id: str, sticker_id: str, **_kwargs: Any) -> str:
    """Get a specific guild sticker."""
    s = _discord_request("GET", f"/guilds/{guild_id}/stickers/{sticker_id}", token)
    return json.dumps({
        "id": s["id"], "name": s["name"], "description": s.get("description"),
        "tags": s.get("tags"), "type": s.get("type"), "available": s.get("available", True),
    })


def _get_sticker(token: str, sticker_id: str, **_kwargs: Any) -> str:
    """Get a sticker by ID from global sticker packs."""
    s = _discord_request("GET", f"/stickers/{sticker_id}", token)
    return json.dumps({"id": s["id"], "name": s["name"], "format_type": s.get("format_type")})


def _list_nitro_sticker_packs(token: str, **_kwargs: Any) -> str:
    """List all standard sticker packs available to Nitro users."""
    packs = _discord_request("GET", "/sticker-packs", token)
    result = []
    for p in packs.get("sticker_packs", []):
        result.append({
            "id": p["id"], "name": p.get("name"),
            "description": p.get("description"),
            "cover_sticker_id": p.get("cover_sticker_id"),
            "stickers": [{"id": s["id"], "name": s["name"]} for s in p.get("stickers", [])[:5]],  # limit context
        })
    return json.dumps({"sticker_packs": result, "count": len(result)})


def _get_auto_moderation_rule(token: str, guild_id: str, rule_id: str, **_kwargs: Any) -> str:
    """Get a specific auto-moderation rule."""
    r = _discord_request("GET", f"/guilds/{guild_id}/auto-moderation/rules/{rule_id}", token)
    return json.dumps({
        "id": r["id"], "name": r.get("name"), "event_type": r.get("event_type"),
        "trigger_type": r.get("trigger_type"), "enabled": r.get("enabled", False),
        "actions": [{"type": a.get("type"), "metadata": a.get("metadata")} for a in r.get("actions", [])],
    })


def _get_guild_scheduled_event_item(token: str, guild_id: str, event_id: str, **_kwargs: Any) -> str:
    """Get a specific scheduled event."""
    params = {}
    with_user_count = _kwargs.get("with_user_count")
    if with_user_count:
        params["with_user_count"] = "true"
    e = _discord_request("GET", f"/guilds/{guild_id}/scheduled-events/{event_id}", token, params=params)
    return json.dumps({
        "id": e["id"], "name": e.get("name"), "description": e.get("description"),
        "scheduled_start_time": e.get("scheduled_start_time"),
        "scheduled_end_time": e.get("scheduled_end_time"),
        "entity_type": e.get("entity_type"), "status": e.get("status"),
        "user_count": e.get("user_count"),
    })


def _get_guild_scheduled_event_users(token: str, guild_id: str, event_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """Get users who RSVP'd to a scheduled event."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params = {"limit": str(min(limit, 100))}
    for key in ("with_member", "before", "after"):
        val = _kwargs.get(key)
        if val is not None:
            params[key] = str(val).lower() if isinstance(val, bool) else val
    users = _discord_request("GET", f"/guilds/{guild_id}/scheduled-events/{event_id}/users", token, params=params)
    result = []
    for u in users:
        user = u.get("user", {})
        result.append({
            "user_id": user.get("id"), "username": user.get("username"),
        })
    return json.dumps({"users": result, "count": len(result)})


def _get_guild_template_by_code(token: str, template_code: str, **_kwargs: Any) -> str:
    """Get a guild template by its code."""
    t = _discord_request("GET", f"/guilds/templates/{template_code}", token)
    return json.dumps({
        "code": t["code"], "name": t.get("name"), "description": t.get("description"),
        "usage_count": t.get("usage_count"),
        "creator_id": t.get("creator_id"),
        "created_at": t.get("created_at"), "updated_at": t.get("updated_at"),
    })


def _create_guild_from_template(token: str, template_code: str, name: str, **_kwargs: Any) -> str:
    """Create a new guild from a template."""
    body: Dict[str, Any] = {"name": name}
    icon = _kwargs.get("icon")
    if icon:
        body["icon"] = icon
    g = _discord_request("POST", f"/guilds/templates/{template_code}", token, body=body)
    return json.dumps({"success": True, "id": g["id"], "name": g.get("name")})


# ---------------------------------------------------------------------------
# User actions
# ---------------------------------------------------------------------------

def _get_current_user(token: str, **_kwargs: Any) -> str:
    """Get the bot user's own info."""
    u = _discord_request("GET", "/users/@me", token)
    return json.dumps({
        "id": u["id"], "username": u.get("username"),
        "discriminator": u.get("discriminator"),
        "global_name": u.get("global_name"),
        "avatar": u.get("avatar"), "bot": u.get("bot", False),
        "flags": u.get("flags", 0),
    })


def _get_user(token: str, user_id: str, **_kwargs: Any) -> str:
    """Get a user by ID."""
    u = _discord_request("GET", f"/users/{user_id}", token)
    return json.dumps({
        "id": u["id"], "username": u.get("username"),
        "global_name": u.get("global_name"),
        "avatar": u.get("avatar"), "bot": u.get("bot", False),
    })


def _modify_current_user(token: str, **_kwargs: Any) -> str:
    """Modify the bot user's profile."""
    body: Dict[str, Any] = {}
    username = _kwargs.get("username")
    if username:
        body["username"] = username
    avatar = _kwargs.get("avatar")
    if avatar:
        body["avatar"] = avatar
    if not body:
        return json.dumps({"error": "No user properties to modify."})
    u = _discord_request("PATCH", "/users/@me", token, body=body)
    return json.dumps({"success": True, "username": u.get("username")})


def _get_current_user_guild_member(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get the bot's guild membership details."""
    m = _discord_request("GET", f"/users/@me/guilds/{guild_id}/member", token)
    return json.dumps({
        "nick": m.get("nick"), "roles": m.get("roles", []),
        "joined_at": m.get("joined_at"), "premium_since": m.get("premium_since"),
    })


# ---------------------------------------------------------------------------
# Application / misc
# ---------------------------------------------------------------------------

def _get_current_application_info(token: str, **_kwargs: Any) -> str:
    """Get the bot application's info."""
    a = _discord_request("GET", "/applications/@me", token)
    return json.dumps({
        "id": a["id"], "name": a.get("name"), "icon": a.get("icon"),
        "description": a.get("description"), "bot_public": a.get("bot_public", False),
        "bot_require_code_grant": a.get("bot_require_code_grant", False),
        "flags": a.get("flags", 0),
    })


def _list_voice_regions(token: str, **_kwargs: Any) -> str:
    """List all available voice regions."""
    regions = _discord_request("GET", "/voice/regions", token)
    return json.dumps({"regions": regions})


# ---------------------------------------------------------------------------
# Action dispatch + metadata
# ---------------------------------------------------------------------------

_ACTIONS = {
    # Existing
    "list_guilds": _list_guilds,
    "server_info": _server_info,
    "list_channels": _list_channels,
    "channel_info": _channel_info,
    "list_roles": _list_roles,
    "member_info": _member_info,
    "search_members": _search_members,
    "fetch_messages": _fetch_messages,
    "list_pins": _list_pins,
    "pin_message": _pin_message,
    "unpin_message": _unpin_message,
    "delete_message": _delete_message,
    "create_thread": _create_thread,
    "add_role": _add_role,
    "remove_role": _remove_role,
    # Channel management
    "create_channel": _create_channel,
    "modify_channel": _modify_channel,
    "delete_channel": _delete_channel,
    # Guild management
    "modify_guild": _modify_guild,
    "get_guild_preview": _get_guild_preview,
    "get_guild_vanity_url": _get_guild_vanity_url,
    "modify_guild_widget": _modify_guild_widget,
    "get_guild_widget_settings": _get_guild_widget_settings,
    "get_guild_widget": _get_guild_widget,
    "list_guild_members": _list_guild_members,
    "modify_guild_member": _modify_guild_member,
    "kick_guild_member": _kick_guild_member,
    "leave_guild": _leave_guild,
    # Ban management
    "list_guild_bans": _list_guild_bans,
    "get_guild_ban": _get_guild_ban,
    "ban_user": _ban_user,
    "unban_user": _unban_user,
    # Role CRUD
    "create_guild_role": _create_guild_role,
    "modify_guild_role": _modify_guild_role,
    "modify_guild_role_positions": _modify_guild_role_positions,
    "delete_guild_role": _delete_guild_role,
    # Emoji management
    "list_guild_emojis": _list_guild_emojis,
    "create_guild_emoji": _create_guild_emoji,
    "modify_guild_emoji": _modify_guild_emoji,
    "delete_guild_emoji": _delete_guild_emoji,
    # Sticker management
    "list_guild_stickers": _list_guild_stickers,
    "create_guild_sticker": _create_guild_sticker,
    "modify_guild_sticker": _modify_guild_sticker,
    "delete_guild_sticker": _delete_guild_sticker,
    # Webhook management
    "list_channel_webhooks": _list_channel_webhooks,
    "list_guild_webhooks": _list_guild_webhooks,
    "create_webhook": _create_webhook,
    "get_webhook": _get_webhook,
    "modify_webhook": _modify_webhook,
    "delete_webhook": _delete_webhook,
    # Thread management
    "list_active_threads": _list_active_threads,
    "list_public_archived_threads": _list_public_archived_threads,
    "list_private_archived_threads": _list_private_archived_threads,
    "list_thread_members": _list_thread_members,
    "manage_thread_member": _manage_thread_member,
    "modify_thread": _modify_thread,
    # Message moderation
    "bulk_delete_messages": _bulk_delete_messages,
    "crosspost_message": _crosspost_message,
    # Auto-moderation
    "list_auto_mod_rules": _list_auto_mod_rules,
    "create_auto_mod_rule": _create_auto_mod_rule,
    "modify_auto_mod_rule": _modify_auto_mod_rule,
    "delete_auto_mod_rule": _delete_auto_mod_rule,
    # Scheduled events
    "list_scheduled_events": _list_scheduled_events,
    "create_scheduled_event": _create_scheduled_event,
    "modify_scheduled_event": _modify_scheduled_event,
    "delete_scheduled_event": _delete_scheduled_event,
    # Invite management
    "create_channel_invite": _create_channel_invite,
    "delete_invite": _delete_invite,
    # Guild templates
    "list_guild_templates": _list_guild_templates,
    "create_guild_template": _create_guild_template,
    "modify_guild_template": _modify_guild_template,
    "delete_guild_template": _delete_guild_template,
    "sync_guild_template": _sync_guild_template,
    # More guild
    "get_guild_audit_log": _get_guild_audit_log,
    "get_guild_welcome_screen": _get_guild_welcome_screen,
    "modify_guild_welcome_screen": _modify_guild_welcome_screen,
    "get_guild_onboarding": _get_guild_onboarding,
    "modify_guild_onboarding": _modify_guild_onboarding,
    "get_guild_voice_regions": _get_guild_voice_regions,
    "get_guild_integrations": _get_guild_integrations,
    "delete_guild_integration": _delete_guild_integration,
    "sync_guild_integration": _sync_guild_integration,
    "get_guild_prune_count": _get_guild_prune_count,
    "begin_guild_prune": _begin_guild_prune,
    "modify_guild_mfa_level": _modify_guild_mfa_level,
    "modify_user_voice_state": _modify_user_voice_state,
    "modify_current_user_voice_state": _modify_current_user_voice_state,
    "delete_guild": _delete_guild,
    "create_guild": _create_guild,
    # More channel
    "modify_guild_channel_positions": _modify_guild_channel_positions,
    "edit_channel_permissions": _edit_channel_permissions,
    "delete_channel_permission": _delete_channel_permission,
    "get_channel_invites": _get_channel_invites,
    "follow_announcement_channel": _follow_announcement_channel,
    "get_invite": _get_invite,
    # Message actions
    "get_channel_message": _get_channel_message,
    "edit_message": _edit_message,
    "add_reaction": _add_reaction,
    "remove_own_reaction": _remove_own_reaction,
    "remove_user_reaction": _remove_user_reaction,
    "get_reactions": _get_reactions,
    "delete_all_reactions": _delete_all_reactions,
    "delete_all_reactions_for_emoji": _delete_all_reactions_for_emoji,
    # Thread more
    "join_thread": _join_thread,
    "leave_thread": _leave_thread,
    "get_thread_member": _get_thread_member,
    "list_joined_private_archived_threads": _list_joined_private_archived_threads,
    # Member
    "modify_current_member": _modify_current_member,
    "add_guild_member": _add_guild_member,
    # Single-item reads
    "get_guild_emoji": _get_guild_emoji,
    "get_guild_sticker": _get_guild_sticker,
    "get_sticker": _get_sticker,
    "list_nitro_sticker_packs": _list_nitro_sticker_packs,
    "get_auto_moderation_rule": _get_auto_moderation_rule,
    "get_guild_scheduled_event_item": _get_guild_scheduled_event_item,
    "get_guild_scheduled_event_users": _get_guild_scheduled_event_users,
    "get_guild_template_by_code": _get_guild_template_by_code,
    "create_guild_from_template": _create_guild_from_template,
    # User
    "get_current_user": _get_current_user,
    "get_user": _get_user,
    "modify_current_user": _modify_current_user,
    "get_current_user_guild_member": _get_current_user_guild_member,
    # Application / misc
    "get_current_application_info": _get_current_application_info,
    "list_voice_regions": _list_voice_regions,
}

_CORE_ACTION_NAMES = frozenset({"fetch_messages", "search_members", "create_thread"})
_ADMIN_ACTION_NAMES = frozenset(_ACTIONS.keys()) - _CORE_ACTION_NAMES

_CORE_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _CORE_ACTION_NAMES}
_ADMIN_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _ADMIN_ACTION_NAMES}

# Single-source-of-truth manifest: action → (signature, one-line description).
# Consumed by :func:`_build_schema` so the schema's top-level description
# always matches the registered action set.
_ACTION_MANIFEST: List[Tuple[str, str, str]] = [
    # Info / read
    ("list_guilds", "()", "list servers the bot is in"),
    ("server_info", "(guild_id)", "server details + member counts"),
    ("get_guild_preview", "(guild_id)", "guild preview for lurkable guilds"),
    ("get_guild_vanity_url", "(guild_id)", "vanity URL invite code if set"),
    ("get_guild_widget", "(guild_id)", "guild widget JSON"),
    ("get_guild_widget_settings", "(guild_id)", "guild widget settings"),
    ("list_channels", "(guild_id)", "all channels grouped by category"),
    ("channel_info", "(channel_id)", "single channel details"),
    ("list_roles", "(guild_id)", "roles sorted by position"),
    ("list_guild_members", "(guild_id)", "list all guild members (paginated)"),
    ("member_info", "(guild_id, user_id)", "lookup a specific member"),
    ("search_members", "(guild_id, query)", "find members by name prefix"),
    ("list_guild_bans", "(guild_id)", "list all bans in the guild"),
    ("get_guild_ban", "(guild_id, user_id)", "get a specific ban"),
    ("list_guild_emojis", "(guild_id)", "list all emojis"),
    ("list_guild_stickers", "(guild_id)", "list all stickers"),
    ("list_scheduled_events", "(guild_id)", "list scheduled events"),
    ("list_auto_mod_rules", "(guild_id)", "list auto-moderation rules"),
    ("list_active_threads", "(guild_id)", "list active threads in guild"),
    ("list_channel_webhooks", "(channel_id)", "list webhooks for a channel"),
    ("list_guild_webhooks", "(guild_id)", "list webhooks for a guild"),
    ("get_webhook", "(webhook_id)", "get a webhook by ID"),
    ("list_guild_templates", "(guild_id)", "list guild templates"),
    ("list_thread_members", "(channel_id)", "list members of a thread"),
    # Message actions
    ("fetch_messages", "(channel_id)", "recent messages; optional before/after snowflakes"),
    ("list_pins", "(channel_id)", "pinned messages in a channel"),
    ("pin_message", "(channel_id, message_id)", "pin a message"),
    ("unpin_message", "(channel_id, message_id)", "unpin a message"),
    ("delete_message", "(channel_id, message_id)", "delete a message"),
    ("bulk_delete_messages", "(channel_id, message_ids)", "bulk delete 2-100 messages by comma-separated IDs"),
    ("crosspost_message", "(channel_id, message_id)", "crosspost/publish in announcement channel"),
    # Thread actions
    ("create_thread", "(channel_id, name)", "create a public thread; optional message_id anchor"),
    ("modify_thread", "(channel_id)", "modify thread settings (archived, locked, etc.)"),
    ("manage_thread_member", "(channel_id, user_id)", "add or remove a member from a thread"),
    ("list_public_archived_threads", "(channel_id)", "list public archived threads"),
    ("list_private_archived_threads", "(channel_id)", "list private archived threads"),
    # Channel management
    ("create_channel", "(guild_id, name)", "create a new channel"),
    ("modify_channel", "(channel_id)", "modify channel settings"),
    ("delete_channel", "(channel_id)", "delete/close a channel"),
    ("create_channel_invite", "(channel_id)", "create an invite for a channel"),
    # Guild management
    ("modify_guild", "(guild_id)", "modify guild settings"),
    ("modify_guild_widget", "(guild_id)", "modify guild widget settings"),
    ("modify_guild_member", "(guild_id, user_id)", "modify member nickname, roles, mute, deaf"),
    ("kick_guild_member", "(guild_id, user_id)", "kick a member from the guild"),
    ("ban_user", "(guild_id, user_id)", "ban a user from the guild"),
    ("unban_user", "(guild_id, user_id)", "remove a ban"),
    ("leave_guild", "(guild_id)", "leave a guild (bot removes itself)"),
    # Role CRUD
    ("create_guild_role", "(guild_id, name)", "create a new role"),
    ("modify_guild_role", "(guild_id, role_id)", "modify a role's name, color, permissions"),
    ("modify_guild_role_positions", "(guild_id, role_id, position)", "reorder a role"),
    ("delete_guild_role", "(guild_id, role_id)", "delete a role"),
    # Emoji
    ("create_guild_emoji", "(guild_id, name, image_data)", "create a new emoji (base64 image)"),
    ("modify_guild_emoji", "(guild_id, emoji_id, name)", "modify emoji name/roles"),
    ("delete_guild_emoji", "(guild_id, emoji_id)", "delete an emoji"),
    # Sticker
    ("create_guild_sticker", "(guild_id, name, sticker_tags, image_data)", "create a sticker (base64 image)"),
    ("modify_guild_sticker", "(guild_id, sticker_id)", "modify a sticker"),
    ("delete_guild_sticker", "(guild_id, sticker_id)", "delete a sticker"),
    # Webhook
    ("create_webhook", "(channel_id, name)", "create a webhook"),
    ("modify_webhook", "(webhook_id)", "modify a webhook"),
    ("delete_webhook", "(webhook_id)", "delete a webhook"),
    # Auto-moderation
    ("create_auto_mod_rule", "(guild_id, name, event_type, trigger_type, actions_json)", "create auto-mod rule"),
    ("modify_auto_mod_rule", "(guild_id, rule_id)", "modify auto-mod rule"),
    ("delete_auto_mod_rule", "(guild_id, rule_id)", "delete auto-mod rule"),
    # Scheduled events
    ("create_scheduled_event", "(guild_id, name, scheduled_start_time, entity_type)", "create a scheduled event"),
    ("modify_scheduled_event", "(guild_id, event_id)", "modify a scheduled event"),
    ("delete_scheduled_event", "(guild_id, event_id)", "delete a scheduled event"),
    # Templates
    ("create_guild_template", "(guild_id, name)", "create a guild template"),
    ("modify_guild_template", "(guild_id, template_code)", "modify a guild template"),
    ("delete_guild_template", "(guild_id, template_code)", "delete a guild template"),
    ("sync_guild_template", "(guild_id, template_code)", "sync template to current guild state"),
    # Invite
    ("delete_invite", "(invite_code)", "delete an invite by code"),
    # Misc
    ("add_role", "(guild_id, user_id, role_id)", "assign a role"),
    ("remove_role", "(guild_id, user_id, role_id)", "remove a role"),
    # More guild actions
    ("get_guild_audit_log", "(guild_id)", "get server audit log entries"),
    ("get_guild_welcome_screen", "(guild_id)", "get welcome screen config"),
    ("modify_guild_welcome_screen", "(guild_id)", "modify welcome screen"),
    ("get_guild_onboarding", "(guild_id)", "get onboarding config"),
    ("modify_guild_onboarding", "(guild_id)", "modify server onboarding"),
    ("get_guild_voice_regions", "(guild_id)", "get voice regions for guild"),
    ("get_guild_integrations", "(guild_id)", "list server integrations"),
    ("delete_guild_integration", "(guild_id, integration_id)", "delete an integration"),
    ("sync_guild_integration", "(guild_id, integration_id)", "sync an integration"),
    ("get_guild_prune_count", "(guild_id)", "count members that would be pruned"),
    ("begin_guild_prune", "(guild_id)", "prune inactive members"),
    ("modify_guild_mfa_level", "(guild_id, level)", "change server MFA requirement"),
    ("modify_user_voice_state", "(guild_id, user_id, channel_id)", "move/mute a member in voice"),
    ("modify_current_user_voice_state", "(guild_id, channel_id)", "move bot in voice"),
    ("delete_guild", "(guild_id)", "delete server permanently (owner only)"),
    ("create_guild", "(name)", "create a new server"),
    # Channel positions/permissions
    ("modify_guild_channel_positions", "(guild_id, channel_positions)", "reorder channels"),
    ("edit_channel_permissions", "(channel_id, overwrite_id, allow, deny)", "edit permission overwrite"),
    ("delete_channel_permission", "(channel_id, overwrite_id)", "delete permission overwrite"),
    ("get_channel_invites", "(channel_id)", "list channel invites"),
    ("follow_announcement_channel", "(channel_id, webhook_channel_id)", "follow an announcement channel"),
    ("get_invite", "(invite_code)", "resolve invite details"),
    # Message read/edit
    ("get_channel_message", "(channel_id, message_id)", "get a single message"),
    ("edit_message", "(channel_id, message_id)", "edit a bot message"),
    ("add_reaction", "(channel_id, message_id, emoji)", "add reaction to a message"),
    ("remove_own_reaction", "(channel_id, message_id, emoji)", "remove own reaction"),
    ("remove_user_reaction", "(channel_id, message_id, emoji, user_id)", "remove another user's reaction"),
    ("get_reactions", "(channel_id, message_id, emoji)", "list users who reacted"),
    ("delete_all_reactions", "(channel_id, message_id)", "remove all reactions"),
    ("delete_all_reactions_for_emoji", "(channel_id, message_id, emoji)", "remove all of one emoji reaction"),
    # Thread join/leave/get
    ("join_thread", "(channel_id)", "bot joins a thread"),
    ("leave_thread", "(channel_id)", "bot leaves a thread"),
    ("get_thread_member", "(channel_id, user_id)", "get thread member details"),
    ("list_joined_private_archived_threads", "(channel_id)", "list joined private archived threads"),
    # Member
    ("modify_current_member", "(guild_id, nick)", "change bot's own nickname"),
    ("add_guild_member", "(guild_id, user_id, access_token)", "add user via OAuth2 token"),
    # Single-item reads
    ("get_guild_emoji", "(guild_id, emoji_id)", "get a specific emoji"),
    ("get_guild_sticker", "(guild_id, sticker_id)", "get a specific sticker"),
    ("get_sticker", "(sticker_id)", "get sticker by ID (global packs)"),
    ("list_nitro_sticker_packs", "()", "list Nitro sticker packs"),
    ("get_auto_moderation_rule", "(guild_id, rule_id)", "get auto-mod rule details"),
    ("get_guild_scheduled_event_item", "(guild_id, event_id)", "get event details"),
    ("get_guild_scheduled_event_users", "(guild_id, event_id)", "get event RSVPs"),
    ("get_guild_template_by_code", "(template_code)", "get template by code"),
    ("create_guild_from_template", "(template_code, name)", "create guild from template"),
    # User
    ("get_current_user", "()", "get bot's own user info"),
    ("get_user", "(user_id)", "get a user by ID"),
    ("modify_current_user", "()", "modify bot's username/avatar"),
    ("get_current_user_guild_member", "(guild_id)", "get bot's guild member info"),
    # Application / misc
    ("get_current_application_info", "()", "get bot application info"),
    ("list_voice_regions", "()", "list all voice regions"),
]

# Actions that require the GUILD_MEMBERS privileged intent.
_INTENT_GATED_MEMBERS = frozenset({
    "member_info", "search_members", "list_guild_members",
})

# Per-action required params for runtime validation.
_REQUIRED_PARAMS: Dict[str, List[str]] = {
    # Existing
    "server_info": ["guild_id"],
    "list_channels": ["guild_id"],
    "list_roles": ["guild_id"],
    "member_info": ["guild_id", "user_id"],
    "search_members": ["guild_id", "query"],
    "channel_info": ["channel_id"],
    "fetch_messages": ["channel_id"],
    "list_pins": ["channel_id"],
    "pin_message": ["channel_id", "message_id"],
    "unpin_message": ["channel_id", "message_id"],
    "delete_message": ["channel_id", "message_id"],
    "create_thread": ["channel_id", "name"],
    "add_role": ["guild_id", "user_id", "role_id"],
    "remove_role": ["guild_id", "user_id", "role_id"],
    # Channel management
    "create_channel": ["guild_id", "name"],
    "modify_channel": ["channel_id"],
    "delete_channel": ["channel_id"],
    # Guild management
    "modify_guild": ["guild_id"],
    "get_guild_preview": ["guild_id"],
    "get_guild_vanity_url": ["guild_id"],
    "modify_guild_widget": ["guild_id"],
    "get_guild_widget_settings": ["guild_id"],
    "get_guild_widget": ["guild_id"],
    "list_guild_members": ["guild_id"],
    "modify_guild_member": ["guild_id", "user_id"],
    "kick_guild_member": ["guild_id", "user_id"],
    "leave_guild": ["guild_id"],
    # Bans
    "list_guild_bans": ["guild_id"],
    "get_guild_ban": ["guild_id", "user_id"],
    "ban_user": ["guild_id", "user_id"],
    "unban_user": ["guild_id", "user_id"],
    # Roles
    "create_guild_role": ["guild_id", "name"],
    "modify_guild_role": ["guild_id", "role_id"],
    "modify_guild_role_positions": ["guild_id", "role_id", "position"],
    "delete_guild_role": ["guild_id", "role_id"],
    # Emoji
    "list_guild_emojis": ["guild_id"],
    "create_guild_emoji": ["guild_id", "name", "image_data"],
    "modify_guild_emoji": ["guild_id", "emoji_id", "name"],
    "delete_guild_emoji": ["guild_id", "emoji_id"],
    # Sticker
    "list_guild_stickers": ["guild_id"],
    "create_guild_sticker": ["guild_id", "name", "sticker_tags", "image_data"],
    "modify_guild_sticker": ["guild_id", "sticker_id"],
    "delete_guild_sticker": ["guild_id", "sticker_id"],
    # Webhook
    "list_channel_webhooks": ["channel_id"],
    "list_guild_webhooks": ["guild_id"],
    "create_webhook": ["channel_id", "name"],
    "get_webhook": ["webhook_id"],
    "modify_webhook": ["webhook_id"],
    "delete_webhook": ["webhook_id"],
    # Thread
    "list_active_threads": ["guild_id"],
    "list_public_archived_threads": ["channel_id"],
    "list_private_archived_threads": ["channel_id"],
    "list_thread_members": ["channel_id"],
    "manage_thread_member": ["channel_id", "user_id"],
    "modify_thread": ["channel_id"],
    # Message moderation
    "bulk_delete_messages": ["channel_id", "message_ids"],
    "crosspost_message": ["channel_id", "message_id"],
    # Auto-mod
    "list_auto_mod_rules": ["guild_id"],
    "create_auto_mod_rule": ["guild_id", "name", "event_type", "trigger_type", "actions_json"],
    "modify_auto_mod_rule": ["guild_id", "rule_id"],
    "delete_auto_mod_rule": ["guild_id", "rule_id"],
    # Scheduled events
    "list_scheduled_events": ["guild_id"],
    "create_scheduled_event": ["guild_id", "name", "scheduled_start_time", "entity_type"],
    "modify_scheduled_event": ["guild_id", "event_id"],
    "delete_scheduled_event": ["guild_id", "event_id"],
    # Invites
    "create_channel_invite": ["channel_id"],
    "delete_invite": ["invite_code"],
    # Templates
    "list_guild_templates": ["guild_id"],
    "create_guild_template": ["guild_id", "name"],
    "modify_guild_template": ["guild_id", "template_code"],
    "delete_guild_template": ["guild_id", "template_code"],
    "sync_guild_template": ["guild_id", "template_code"],
    # More guild
    "get_guild_audit_log": ["guild_id"],
    "get_guild_welcome_screen": ["guild_id"],
    "modify_guild_welcome_screen": ["guild_id"],
    "get_guild_onboarding": ["guild_id"],
    "modify_guild_onboarding": ["guild_id"],
    "get_guild_voice_regions": ["guild_id"],
    "get_guild_integrations": ["guild_id"],
    "delete_guild_integration": ["guild_id", "integration_id"],
    "sync_guild_integration": ["guild_id", "integration_id"],
    "get_guild_prune_count": ["guild_id"],
    "begin_guild_prune": ["guild_id"],
    "modify_guild_mfa_level": ["guild_id", "level"],
    "modify_user_voice_state": ["guild_id", "user_id", "channel_id"],
    "modify_current_user_voice_state": ["guild_id", "channel_id"],
    "delete_guild": ["guild_id"],
    "create_guild": ["name"],
    # More channel
    "modify_guild_channel_positions": ["guild_id", "channel_positions"],
    "edit_channel_permissions": ["channel_id", "overwrite_id", "allow", "deny"],
    "delete_channel_permission": ["channel_id", "overwrite_id"],
    "get_channel_invites": ["channel_id"],
    "follow_announcement_channel": ["channel_id", "webhook_channel_id"],
    "get_invite": ["invite_code"],
    # Message
    "get_channel_message": ["channel_id", "message_id"],
    "edit_message": ["channel_id", "message_id"],
    "add_reaction": ["channel_id", "message_id", "emoji"],
    "remove_own_reaction": ["channel_id", "message_id", "emoji"],
    "remove_user_reaction": ["channel_id", "message_id", "emoji", "user_id"],
    "get_reactions": ["channel_id", "message_id", "emoji"],
    "delete_all_reactions": ["channel_id", "message_id"],
    "delete_all_reactions_for_emoji": ["channel_id", "message_id", "emoji"],
    # Thread
    "join_thread": ["channel_id"],
    "leave_thread": ["channel_id"],
    "get_thread_member": ["channel_id", "user_id"],
    "list_joined_private_archived_threads": ["channel_id"],
    # Member
    "modify_current_member": ["guild_id", "nick"],
    "add_guild_member": ["guild_id", "user_id", "access_token"],
    # Single-item reads
    "get_guild_emoji": ["guild_id", "emoji_id"],
    "get_guild_sticker": ["guild_id", "sticker_id"],
    "get_sticker": ["sticker_id"],
    "list_nitro_sticker_packs": [],
    "get_auto_moderation_rule": ["guild_id", "rule_id"],
    "get_guild_scheduled_event_item": ["guild_id", "event_id"],
    "get_guild_scheduled_event_users": ["guild_id", "event_id"],
    "get_guild_template_by_code": ["template_code"],
    "create_guild_from_template": ["template_code", "name"],
    # User
    "get_current_user": [],
    "get_user": ["user_id"],
    "modify_current_user": [],
    "get_current_user_guild_member": ["guild_id"],
    # Application / misc
    "get_current_application_info": [],
    "list_voice_regions": [],
}


# ---------------------------------------------------------------------------
# Config-based action allowlist
# ---------------------------------------------------------------------------

def _load_allowed_actions_config() -> Optional[List[str]]:
    """Read ``discord.server_actions`` from user config.

    Returns a list of allowed action names, or ``None`` if the user
    hasn't restricted the set (default: all actions allowed).

    Accepts either a comma-separated string or a YAML list.
    Unknown action names are dropped with a log warning.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("discord: could not load config (%s); allowing all actions.", exc)
        return None

    raw = (cfg.get("discord") or {}).get("server_actions")
    if raw is None or raw == "":
        return None

    if isinstance(raw, str):
        names = [n.strip() for n in raw.split(",") if n.strip()]
    elif isinstance(raw, (list, tuple)):
        names = [str(n).strip() for n in raw if str(n).strip()]
    else:
        logger.warning(
            "discord.server_actions: unexpected type %s; ignoring.", type(raw).__name__,
        )
        return None

    valid = [n for n in names if n in _ACTIONS]
    invalid = [n for n in names if n not in _ACTIONS]
    if invalid:
        logger.warning(
            "discord.server_actions: unknown action(s) ignored: %s. "
            "Known: %s",
            ", ".join(invalid), ", ".join(_ACTIONS.keys()),
        )
    return valid


def _available_actions(
    caps: Dict[str, Any],
    allowlist: Optional[List[str]],
) -> List[str]:
    """Compute the visible action list from intents + config allowlist.

    Preserves the canonical order from :data:`_ACTIONS`.
    """
    actions: List[str] = []
    for name in _ACTIONS:
        # Intent filter
        if not caps.get("has_members_intent", True) and name in _INTENT_GATED_MEMBERS:
            continue
        # Config allowlist filter
        if allowlist is not None and name not in allowlist:
            continue
        actions.append(name)
    return actions


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def _build_schema(
    actions: List[str],
    caps: Optional[Dict[str, Any]] = None,
    tool_name: str = "discord",
) -> Optional[Dict[str, Any]]:
    """Build the tool schema for the given filtered action list.

    Returns ``None`` when *actions* is empty — callers should drop the
    tool from registration in that case.
    """
    caps = caps or {}
    if not actions:
        return None

    # Action manifest lines (action-first, parameter-scoped).
    manifest_lines = [
        f"  {name}{sig}  — {desc}"
        for name, sig, desc in _ACTION_MANIFEST
        if name in actions
    ]
    manifest_block = "\n".join(manifest_lines)

    content_note = ""
    affected_actions = {"fetch_messages", "list_pins"} & set(actions)
    if affected_actions and caps.get("detected") and caps.get("has_message_content") is False:
        names = " and ".join(sorted(affected_actions))
        content_note = (
            f"\n\nNOTE: Bot does NOT have the MESSAGE_CONTENT privileged intent. "
            f"{names} will return message metadata (author, "
            "timestamps, attachments, reactions, pin state) but `content` will be "
            "empty for messages not sent as a direct mention to the bot or in DMs. "
            "Enable the intent in the Discord Developer Portal to see all content."
        )

    if tool_name == "discord_admin":
        description = (
            "Manage a Discord server via the REST API.\n\n"
            "Available actions:\n"
            f"{manifest_block}\n\n"
            "Call list_guilds first to discover guild_ids, then list_channels for "
            "channel_ids. Runtime errors will tell you if the bot lacks a specific "
            "per-guild permission (e.g. MANAGE_ROLES for add_role)."
            f"{content_note}"
        )
    else:
        description = (
            "Read and participate in a Discord server.\n\n"
            "Available actions:\n"
            f"{manifest_block}\n\n"
            "Use the channel_id from the current conversation context. "
            "Use search_members to look up user IDs by name prefix."
            f"{content_note}"
        )

    properties: Dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": actions,
        },
        "guild_id": {
            "type": "string",
            "description": "Discord server (guild) ID.",
        },
        "channel_id": {
            "type": "string",
            "description": "Discord channel ID.",
        },
        "user_id": {
            "type": "string",
            "description": "Discord user ID.",
        },
        "role_id": {
            "type": "string",
            "description": "Discord role ID.",
        },
        "message_id": {
            "type": "string",
            "description": "Discord message ID.",
        },
        "query": {
            "type": "string",
            "description": "Member name prefix to search for (search_members).",
        },
        "name": {
            "type": "string",
            "description": "Name for threads, channels, roles, webhooks, emojis, stickers, templates, events.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Max results (default 50). Applies to fetch_messages, search_members, list_*. May be capped by API endpoint.",
        },
        "before": {
            "type": "string",
            "description": "Snowflake ID for reverse pagination / archive listing.",
        },
        "after": {
            "type": "string",
            "description": "Snowflake ID for forward pagination.",
        },
        "auto_archive_duration": {
            "type": "integer",
            "enum": [60, 1440, 4320, 10080],
            "description": "Thread archive duration in minutes (create_thread, modify_thread, default 1440).",
        },
        # -- New universal/common params --
        "reason": {
            "type": "string",
            "description": "Audit log reason shown in Discord server logs (kick, ban, modify, etc.).",
        },
        "params_json": {
            "type": "string",
            "description": "Additional JSON body params for advanced/complex modifications (modify_channel, modify_guild, etc.). Merged into the API request body.",
        },
        "position": {
            "type": "integer",
            "description": "Channel or role position (modify_channel_role_positions, create_channel).",
        },
        "topic": {
            "type": "string",
            "description": "Channel topic (create_channel, modify_channel).",
        },
        "nsfw": {
            "type": "boolean",
            "description": "Whether the channel is NSFW (create_channel, modify_channel).",
        },
        "bitrate": {
            "type": "integer",
            "minimum": 8000,
            "maximum": 384000,
            "description": "Voice channel bitrate (create_channel, modify_channel).",
        },
        "user_limit": {
            "type": "integer",
            "minimum": 0,
            "maximum": 99,
            "description": "Voice channel user limit (create_channel, modify_channel, 0=unlimited).",
        },
        "rate_limit_per_user": {
            "type": "integer",
            "minimum": 0,
            "maximum": 21600,
            "description": "Slow mode cooldown in seconds (channel/thread create/modify).",
        },
        "channel_type": {
            "type": "integer",
            "enum": [0, 2, 4, 5, 13, 14, 15],
            "description": "Channel type: 0=text, 2=voice, 4=category, 5=announcement, 13=stage, 14=directory, 15=forum (create_channel).",
        },
        "parent_id": {
            "type": "string",
            "description": "Parent category ID for nesting channels under a category.",
        },
        "permission_overwrites": {
            "type": "string",
            "description": "JSON array of permission overwrite objects (create_channel, modify_channel).",
        },
        "nick": {
            "type": "string",
            "description": "Member nickname (modify_guild_member).",
        },
        "roles": {
            "type": "string",
            "description": "Comma-separated role IDs (modify_guild_member, create_emoji, modify_emoji).",
        },
        "mute": {
            "type": "boolean",
            "description": "Whether to mute a member in voice (modify_guild_member).",
        },
        "deaf": {
            "type": "boolean",
            "description": "Whether to deafen a member in voice (modify_guild_member).",
        },
        "communication_disabled_until": {
            "type": "string",
            "description": "ISO8601 timestamp until member is timed out (modify_guild_member).",
        },
        "delete_message_days": {
            "type": "integer",
            "minimum": 0,
            "maximum": 7,
            "description": "Days of messages to delete when banning (ban_user, default 0).",
        },
        "delete_message_seconds": {
            "type": "integer",
            "minimum": 0,
            "maximum": 604800,
            "description": "Seconds of messages to delete when banning (ban_user, modern API).",
        },
        "color": {
            "type": "integer",
            "description": "Role color as an integer (create_role, modify_role).",
        },
        "hoist": {
            "type": "boolean",
            "description": "Show role separately in the sidebar (create_role, modify_role).",
        },
        "mentionable": {
            "type": "boolean",
            "description": "Allow role to be @mentioned by anyone (create_role, modify_role).",
        },
        "permissions": {
            "type": "string",
            "description": "Permission bit-set as a string (create_role, modify_role).",
        },
        "icon": {
            "type": "string",
            "description": "Role icon URL or base64 data URI (create_role, modify_role).",
        },
        "unicode_emoji": {
            "type": "string",
            "description": "Role unicode emoji (create_role, modify_role).",
        },
        "image_data": {
            "type": "string",
            "description": "Base64-encoded image data (create_guild_emoji, create_guild_sticker).",
        },
        "emoji_id": {
            "type": "string",
            "description": "Emoji ID (modify_emoji, delete_emoji).",
        },
        "sticker_tags": {
            "type": "string",
            "description": "Autocomplete tags for stickers (create_sticker, modify_sticker).",
        },
        "sticker_description": {
            "type": "string",
            "description": "Sticker description (create_sticker, modify_sticker).",
        },
        "sticker_id": {
            "type": "string",
            "description": "Sticker ID (modify_sticker, delete_sticker).",
        },
        "webhook_id": {
            "type": "string",
            "description": "Webhook ID (get_webhook, modify_webhook, delete_webhook).",
        },
        "webhook_avatar": {
            "type": "string",
            "description": "Webhook avatar image data URI or URL (create_webhook, modify_webhook).",
        },
        "archived": {
            "type": "boolean",
            "description": "Whether a thread is archived (modify_thread).",
        },
        "locked": {
            "type": "boolean",
            "description": "Whether a thread is locked (modify_thread).",
        },
        "invitable": {
            "type": "boolean",
            "description": "Whether non-moderators can invite to a private thread (modify_thread).",
        },
        "message_ids": {
            "type": "string",
            "description": "Comma-separated message IDs for bulk delete (2-100 IDs).",
        },
        "invite_max_age": {
            "type": "integer",
            "description": "Invite max age in seconds, 0=never (create_channel_invite, default 86400).",
        },
        "invite_max_uses": {
            "type": "integer",
            "description": "Invite max uses, 0=unlimited (create_channel_invite, default 0).",
        },
        "invite_temporary": {
            "type": "boolean",
            "description": "Grant temporary membership (kicked on disconnect) (create_channel_invite).",
        },
        "invite_unique": {
            "type": "boolean",
            "description": "Create a unique invite even if one already exists (create_channel_invite).",
        },
        "invite_code": {
            "type": "string",
            "description": "Invite code (delete_invite).",
        },
        "event_id": {
            "type": "string",
            "description": "Scheduled event ID (modify_event, delete_event).",
        },
        "event_description": {
            "type": "string",
            "description": "Scheduled event description (create_event, modify_event).",
        },
        "scheduled_start_time": {
            "type": "string",
            "description": "ISO 8601 event start time (create_scheduled_event, modify_scheduled_event).",
        },
        "scheduled_end_time": {
            "type": "string",
            "description": "ISO 8601 event end time (create_scheduled_event, modify_scheduled_event).",
        },
        "entity_type": {
            "type": "integer",
            "enum": [1, 2, 3],
            "description": "Entity type: 1=stage instance, 2=voice, 3=external (create_scheduled_event).",
        },
        "entity_metadata": {
            "type": "string",
            "description": "JSON entity metadata for external events (e.g., location).",
        },
        "privacy_level": {
            "type": "integer",
            "description": "Event privacy level: 2=guild only (scheduled events).",
        },
        "status": {
            "type": "integer",
            "description": "Event status: 1=scheduled, 2=active, 3=completed, 4=cancelled (modify_event).",
        },
        "with_user_count": {
            "type": "boolean",
            "description": "Include user count in scheduled event list (list_scheduled_events).",
        },
        "rule_id": {
            "type": "string",
            "description": "Auto-moderation rule ID (modify_rule, delete_rule).",
        },
        "event_type": {
            "type": "integer",
            "description": "Auto-mod event type: 1=message_send (create_rule, modify_rule).",
        },
        "trigger_type": {
            "type": "integer",
            "description": "Auto-mod trigger type: 1=keyword, 2=harmful_link, 3=spam, 4=keyword_preset, 5=mention_spam (create_rule).",
        },
        "actions_json": {
            "type": "string",
            "description": "JSON array of auto-mod action objects (create_rule, modify_rule).",
        },
        "keyword_filter": {
            "type": "string",
            "description": "Comma-separated keyword filter for auto-mod rules (create_rule, modify_rule).",
        },
        "trigger_metadata": {
            "type": "string",
            "description": "JSON trigger metadata for auto-mod rules (create_rule, modify_rule).",
        },
        "enabled": {
            "type": "boolean",
            "description": "Whether the feature/rule is enabled (create_rule, modify_rule, modify_widget, modify_guild).",
        },
        "exempt_roles": {
            "type": "string",
            "description": "Comma-separated role IDs exempt from auto-mod rule (create_rule, modify_rule).",
        },
        "exempt_channels": {
            "type": "string",
            "description": "Comma-separated channel IDs exempt from auto-mod rule (create_rule, modify_rule).",
        },
        "modify_guild_description": {
            "type": "string",
            "description": "Guild description (modify_guild).",
        },
        "verification_level": {
            "type": "integer",
            "enum": [0, 1, 2, 3, 4],
            "description": "Guild verification level: 0=none, 1=low, 2=medium, 3=high, 4=very_high (modify_guild).",
        },
        "template_code": {
            "type": "string",
            "description": "Guild template code (modify_template, delete_template, sync_template).",
        },
        "template_description": {
            "type": "string",
            "description": "Guild template description (create_template, modify_template).",
        },
    }

    return {
        "name": tool_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["action"],
        },
    }


def _get_dynamic_schema(
    action_subset: Dict[str, Any],
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    """Build a dynamic schema for *action_subset* filtered by intents + config."""
    token = _get_bot_token()
    if not token:
        return None
    caps = _detect_capabilities(token)
    allowlist = _load_allowed_actions_config()
    actions = [a for a in _available_actions(caps, allowlist) if a in action_subset]
    if not actions:
        return None
    return _build_schema(actions, caps, tool_name=tool_name)


def get_dynamic_schema_core() -> Optional[Dict[str, Any]]:
    return _get_dynamic_schema(_CORE_ACTIONS, "discord")


def get_dynamic_schema_admin() -> Optional[Dict[str, Any]]:
    return _get_dynamic_schema(_ADMIN_ACTIONS, "discord_admin")


def get_dynamic_schema() -> Optional[Dict[str, Any]]:
    """Backward-compat wrapper — returns core schema."""
    return get_dynamic_schema_core()


# ---------------------------------------------------------------------------
# 403 error enrichment
# ---------------------------------------------------------------------------

_ACTION_403_HINT = {
    # Existing
    "pin_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel. "
        "Ask the server admin to grant the bot a role that has MANAGE_MESSAGES, "
        "or a per-channel overwrite."
    ),
    "unpin_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel."
    ),
    "delete_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel, or cannot view the channel/message."
    ),
    "create_thread": (
        "Bot lacks CREATE_PUBLIC_THREADS in this channel, or cannot view it."
    ),
    "add_role": (
        "Either the bot lacks MANAGE_ROLES, or the target role sits higher "
        "than the bot's highest role. Roles can only be assigned below the "
        "bot's own position in the role hierarchy."
    ),
    "remove_role": (
        "Either the bot lacks MANAGE_ROLES, or the target role sits higher "
        "than the bot's highest role."
    ),
    "fetch_messages": (
        "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
    ),
    "list_pins": (
        "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
    ),
    "channel_info": (
        "Bot cannot view this channel (missing VIEW_CHANNEL)."
    ),
    "search_members": (
        "Likely missing the Server Members privileged intent — enable it in the "
        "Discord Developer Portal under your bot's settings."
    ),
    "member_info": (
        "Bot cannot see this guild member (missing Server Members intent or "
        "insufficient permissions)."
    ),
    # New channel actions
    "create_channel": (
        "Bot lacks MANAGE_CHANNELS permission in this guild."
    ),
    "modify_channel": (
        "Bot lacks MANAGE_CHANNELS permission in this channel/guild."
    ),
    "delete_channel": (
        "Bot lacks MANAGE_CHANNELS permission in this channel/guild."
    ),
    "create_channel_invite": (
        "Bot lacks CREATE_INSTANT_INVITE permission in this channel."
    ),
    # New guild actions
    "modify_guild": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "modify_guild_widget": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "modify_guild_member": (
        "Bot lacks MODERATE_MEMBERS or MANAGE_NICKNAMES permission, or the target role "
        "sits higher than the bot's highest role."
    ),
    "kick_guild_member": (
        "Bot lacks KICK_MEMBERS permission, or the target member's highest role "
        "sits above the bot's highest role."
    ),
    # Ban
    "ban_user": (
        "Bot lacks BAN_MEMBERS permission, or the target user's highest role "
        "sits above the bot's highest role."
    ),
    "unban_user": (
        "Bot lacks BAN_MEMBERS permission."
    ),
    "list_guild_bans": (
        "Bot lacks BAN_MEMBERS permission."
    ),
    "get_guild_ban": (
        "Bot lacks BAN_MEMBERS permission."
    ),
    # Role CRUD
    "create_guild_role": (
        "Bot lacks MANAGE_ROLES permission."
    ),
    "modify_guild_role": (
        "Bot lacks MANAGE_ROLES permission, or the target role sits higher "
        "than the bot's highest role."
    ),
    "modify_guild_role_positions": (
        "Bot lacks MANAGE_ROLES permission."
    ),
    "delete_guild_role": (
        "Bot lacks MANAGE_ROLES permission, or the target role sits higher "
        "than the bot's highest role."
    ),
    # Emoji
    "create_guild_emoji": (
        "Bot lacks MANAGE_GUILD_EXPRESSIONS (or MANAGE_EMOJIS) permission."
    ),
    "modify_guild_emoji": (
        "Bot lacks MANAGE_GUILD_EXPRESSIONS (or MANAGE_EMOJIS) permission."
    ),
    "delete_guild_emoji": (
        "Bot lacks MANAGE_GUILD_EXPRESSIONS (or MANAGE_EMOJIS) permission."
    ),
    # Sticker
    "create_guild_sticker": (
        "Bot lacks MANAGE_GUILD_EXPRESSIONS (or MANAGE_EMOJIS_AND_STICKERS) permission."
    ),
    "modify_guild_sticker": (
        "Bot lacks MANAGE_GUILD_EXPRESSIONS permission."
    ),
    "delete_guild_sticker": (
        "Bot lacks MANAGE_GUILD_EXPRESSIONS permission."
    ),
    # Webhook
    "create_webhook": (
        "Bot lacks MANAGE_WEBHOOKS permission in this channel."
    ),
    "modify_webhook": (
        "Bot lacks MANAGE_WEBHOOKS permission."
    ),
    "delete_webhook": (
        "Bot lacks MANAGE_WEBHOOKS permission."
    ),
    "list_channel_webhooks": (
        "Bot lacks MANAGE_WEBHOOKS permission in this channel."
    ),
    "list_guild_webhooks": (
        "Bot lacks MANAGE_WEBHOOKS permission in this guild."
    ),
    # Thread
    "modify_thread": (
        "Bot lacks MANAGE_THREADS permission or cannot view this thread."
    ),
    "manage_thread_member": (
        "Bot lacks MANAGE_THREADS permission, or the thread is private and the bot isn't a member."
    ),
    # Moderation
    "bulk_delete_messages": (
        "Bot lacks MANAGE_MESSAGES permission in this channel."
    ),
    "crosspost_message": (
        "Bot lacks SEND_MESSAGES or READ_MESSAGE_HISTORY in this announcement channel."
    ),
    # Auto-mod
    "create_auto_mod_rule": (
        "Bot lacks MANAGE_GUILD permission to create auto-mod rules."
    ),
    "modify_auto_mod_rule": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "delete_auto_mod_rule": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "list_auto_mod_rules": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    # Scheduled events
    "create_scheduled_event": (
        "Bot lacks MANAGE_EVENTS permission."
    ),
    "modify_scheduled_event": (
        "Bot lacks MANAGE_EVENTS permission."
    ),
    "delete_scheduled_event": (
        "Bot lacks MANAGE_EVENTS permission."
    ),
    # Invite
    "delete_invite": (
        "Bot lacks MANAGE_CHANNELS or MANAGE_GUILD permission."
    ),
    # Templates
    "create_guild_template": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "modify_guild_template": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "delete_guild_template": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "sync_guild_template": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    # Misc
    "list_guild_members": (
        "Likely missing the Server Members privileged intent — enable it in the "
        "Discord Developer Portal. Also requires VIEW_CHANNEL on at least one channel."
    ),
    "leave_guild": (
        "Bot cannot leave the guild via this token setup; the bot was likely added via OAuth2."
    ),
    # More guild
    "get_guild_audit_log": (
        "Bot lacks VIEW_AUDIT_LOG permission."
    ),
    "modify_guild_welcome_screen": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "modify_guild_onboarding": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "get_guild_integrations": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "delete_guild_integration": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "sync_guild_integration": (
        "Bot lacks MANAGE_GUILD permission."
    ),
    "begin_guild_prune": (
        "Bot lacks KICK_MEMBERS permission."
    ),
    "modify_user_voice_state": (
        "Bot lacks MUTE_MEMBERS or DEAFEN_MEMBERS permission, or the member is in a different channel."
    ),
    "delete_guild": (
        "Bot must be the owner of the guild to delete it."
    ),
    # Permission overwrites
    "edit_channel_permissions": (
        "Bot lacks MANAGE_ROLES permission on this channel."
    ),
    "delete_channel_permission": (
        "Bot lacks MANAGE_ROLES permission on this channel."
    ),
    # Messages
    "add_reaction": (
        "Bot lacks ADD_REACTIONS or READ_MESSAGE_HISTORY in this channel."
    ),
    "remove_own_reaction": (
        "Bot lacks READ_MESSAGE_HISTORY in this channel."
    ),
    "remove_user_reaction": (
        "Bot lacks MANAGE_MESSAGES in this channel."
    ),
    "delete_all_reactions": (
        "Bot lacks MANAGE_MESSAGES in this channel."
    ),
    "edit_message": (
        "Bot can only edit its own messages, or lacks MANAGE_MESSAGES."
    ),
    # Thread
    "join_thread": (
        "Bot cannot view or join this thread (missing permissions or private thread)."
    ),
    "leave_thread": (
        "Bot is not a member of this thread."
    ),
    # Member
    "modify_current_member": (
        "Bot lacks CHANGE_NICKNAME permission."
    ),
    "add_guild_member": (
        "Missing guilds.join OAuth2 scope or the user is already in the guild."
    ),
    # Guild from template
    "create_guild_from_template": (
        "Bot is rate-limited (only 1 guild creation per 10 minutes) or in 10+ guilds."
    ),
    # User
    "modify_current_user": (
        "Bot token cannot modify user profile; use OAuth2 user token instead."
    ),
}


def _enrich_403(action: str, body: str) -> str:
    """Return a user-friendly guidance string for a 403 on ``action``."""
    hint = _ACTION_403_HINT.get(action)
    base = f"Discord API 403 (forbidden) on '{action}'."
    if hint:
        return f"{base} {hint} (Raw: {body})"
    return f"{base} (Raw: {body})"


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def check_discord_tool_requirements() -> bool:
    """Tool is available only when a Discord bot token is configured."""
    return bool(_get_bot_token())


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _run_discord_action(
    action: str,
    valid_actions: Dict[str, Any],
    tool_label: str,
    **kwargs,
) -> str:
    """Shared handler logic for both discord tools."""
    token = _get_bot_token()
    if not token:
        return json.dumps({"error": "DISCORD_BOT_TOKEN not configured."})

    action_fn = valid_actions.get(action)
    if not action_fn:
        return json.dumps({
            "error": f"Unknown action: {action}",
            "available_actions": list(valid_actions.keys()),
        })

    # Config-level allowlist gate (defense in depth — schema already filtered,
    # but a stale cached schema from a prior config should not let denied
    # actions through).
    allowlist = _load_allowed_actions_config()
    if allowlist is not None and action not in allowlist:
        return json.dumps({
            "error": (
                f"Action '{action}' is disabled by config (discord.server_actions). "
                f"Allowed: {', '.join(allowlist) if allowlist else '<none>'}"
            ),
        })

    # Build required-params check from kwargs
    missing = [p for p in _REQUIRED_PARAMS.get(action, []) if not kwargs.get(p)]
    if missing:
        return json.dumps({
            "error": f"Missing required parameters for '{action}': {', '.join(missing)}",
        })

    try:
        # Forward ALL kwargs to the action function — each picks what it needs via **_kwargs
        return action_fn(token=token, **kwargs)
    except DiscordAPIError as e:
        logger.warning("Discord API error in %s action '%s': %s", tool_label, action, e)
        if e.status == 403:
            return json.dumps({"error": _enrich_403(action, e.body)})
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception("Unexpected error in %s action '%s'", tool_label, action)
        return json.dumps({"error": f"Unexpected error: {e}"})


def discord_core(action: str, **kwargs) -> str:
    """Execute a core Discord action (fetch_messages, search_members, create_thread)."""
    return _run_discord_action(action, _CORE_ACTIONS, "discord", **kwargs)


def discord_admin_handler(action: str, **kwargs) -> str:
    """Execute a Discord admin action (server management)."""
    return _run_discord_action(action, _ADMIN_ACTIONS, "discord_admin", **kwargs)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

_HANDLER_DEFAULTS = {
    "action": "", "guild_id": "", "channel_id": "", "user_id": "",
    "role_id": "", "message_id": "", "query": "", "name": "",
    "limit": 50, "before": "", "after": "", "auto_archive_duration": 1440,
    "position": 0, "topic": "", "nsfw": False, "bitrate": 64000,
    "user_limit": 0, "rate_limit_per_user": 0, "channel_type": 0,
    "parent_id": "", "permission_overwrites": "", "reason": "",
    "params_json": "", "nick": "", "roles": "", "mute": False,
    "deaf": False, "communication_disabled_until": "",
    "delete_message_days": 0, "delete_message_seconds": 0,
    "color": 0, "hoist": False, "mentionable": False, "permissions": "",
    "icon": "", "unicode_emoji": "", "image_data": "", "emoji_id": "",
    "sticker_tags": "", "sticker_description": "", "sticker_id": "",
    "webhook_id": "", "webhook_avatar": "", "archived": False,
    "locked": False, "invitable": False, "message_ids": "",
    "invite_max_age": 86400, "invite_max_uses": 0, "invite_temporary": False,
    "invite_unique": False, "invite_code": "",
    "event_id": "", "event_description": "", "scheduled_start_time": "",
    "scheduled_end_time": "", "entity_type": 2, "entity_metadata": "",
    "privacy_level": 2, "status": 1, "with_user_count": False,
    "rule_id": "", "event_type": 1, "trigger_type": 1, "actions_json": "",
    "keyword_filter": "", "trigger_metadata": "", "enabled": True,
    "exempt_roles": "", "exempt_channels": "",
    "modify_guild_description": "", "verification_level": 0,
    "template_code": "", "template_description": "",
}


def _make_handler(handler_fn):
    """Create a registry-compatible handler lambda for a discord handler."""
    return lambda args, **kw: handler_fn(
        **{k: args.get(k, v) for k, v in _HANDLER_DEFAULTS.items()},
    )


_STATIC_CORE_SCHEMA = _build_schema(
    list(_CORE_ACTIONS.keys()), caps={"detected": False}, tool_name="discord",
)
_STATIC_ADMIN_SCHEMA = _build_schema(
    list(_ADMIN_ACTIONS.keys()), caps={"detected": False}, tool_name="discord_admin",
)

registry.register(
    name="discord",
    toolset="discord",
    schema=_STATIC_CORE_SCHEMA,
    handler=_make_handler(discord_core),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)

registry.register(
    name="discord_admin",
    toolset="discord_admin",
    schema=_STATIC_ADMIN_SCHEMA,
    handler=_make_handler(discord_admin_handler),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)
