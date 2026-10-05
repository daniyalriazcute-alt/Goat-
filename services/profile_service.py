from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import re

try:
    from firebase_service import _db
except Exception:
    _db = None


# ============================================================
# PROGRESS WEIGHTS
# ============================================================

PROGRESS_KEYS = {
    "chat": 10,
    "roadmap": 20,
    "skills": 20,
    "opportunities": 15,
    "resources": 15,
    "goal": 10,
    "profile": 10,
}


# ============================================================
# TIME
# ============================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# FIRESTORE CLIENT
# ============================================================

def _get_db():
    """
    Return the same Firestore Admin client used by
    the rest of the AuraAI application.
    """

    if _db is None:
        raise RuntimeError(
            "firebase_service._db could not be imported."
        )

    try:
        db = _db()
    except Exception as exc:
        raise RuntimeError(
            "Firebase Admin Firestore initialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if db is None:
        raise RuntimeError(
            "Firestore client is None. "
            "Check the [firebase] section in Streamlit Secrets."
        )

    return db


# ============================================================
# DEFAULT PROFILE
# ============================================================

def _defaults(
    user: dict[str, Any],
) -> dict[str, Any]:

    now = _now()

    return {
        "uid": user.get("uid", ""),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "career_goal": "",
        "completed": [],
        "progress": 0,
        "recent_chats": [],
        "created_at": now,
        "updated_at": now,
    }


# ============================================================
# PROGRESS
# ============================================================

def _calculate_progress(
    completed: list[str],
) -> int:

    return min(
        100,
        sum(
            PROGRESS_KEYS.get(item, 0)
            for item in set(completed)
        ),
    )


# ============================================================
# GET PROFILE
# ============================================================

def get_profile(
    user: dict[str, Any],
) -> dict[str, Any]:

    profile = _defaults(user)

    uid = user.get("uid")

    if uid:

        try:

            db = _get_db()

            snapshot = (
                db.collection("users")
                .document(uid)
                .get()
            )

            if snapshot.exists:

                data = (
                    snapshot.to_dict()
                    or {}
                )

                profile.update(data)

        except Exception:
            # Do not break AuraAI if profile loading fails.
            pass

    completed = list(
        profile.get("completed") or []
    )

    profile["completed"] = completed

    profile["progress"] = _calculate_progress(
        completed
    )

    return profile


# ============================================================
# MARK COMPLETE
# ============================================================

def mark_complete(
    user: dict[str, Any],
    item: str,
) -> dict[str, Any]:

    item = str(item).strip()

    profile = get_profile(user)

    if not item:
        return profile

    completed = set(
        profile.get("completed") or []
    )

    completed.add(item)

    completed_list = sorted(completed)

    progress = _calculate_progress(
        completed_list
    )

    updated_at = _now()

    profile["completed"] = completed_list
    profile["progress"] = progress
    profile["updated_at"] = updated_at

    uid = user.get("uid")

    if uid:

        try:

            db = _get_db()

            (
                db.collection("users")
                .document(uid)
                .set(
                    {
                        "completed": completed_list,
                        "progress": progress,
                        "updated_at": updated_at,
                    },
                    merge=True,
                )
            )

        except Exception:
            pass

    return profile


# ============================================================
# SET CAREER GOAL
# ============================================================

def set_goal(
    user: dict[str, Any],
    goal: str,
) -> dict[str, Any]:

    clean_goal = (
        str(goal)
        .strip()
        [:500]
    )

    profile = get_profile(user)

    profile["career_goal"] = clean_goal
    profile["updated_at"] = _now()

    uid = user.get("uid")

    if uid:

        try:

            db = _get_db()

            (
                db.collection("users")
                .document(uid)
                .set(
                    {
                        "career_goal": clean_goal,
                        "updated_at": profile[
                            "updated_at"
                        ],
                    },
                    merge=True,
                )
            )

        except Exception:
            pass

    return mark_complete(
        user,
        "goal",
    )


# ============================================================
# SAVE RECENT CHAT
# ============================================================

def save_recent_chat(
    user: dict[str, Any],
    title: str,
    prompt: str,
) -> dict[str, Any]:

    profile = get_profile(user)

    new_chat = {
        "title": str(title).strip()[:120],
        "prompt": str(prompt).strip()[:1000],
        "timestamp": _now(),
    }

    existing_chats = list(
        profile.get("recent_chats") or []
    )

    chats = [
        new_chat,
        *existing_chats,
    ]

    chats = chats[:8]

    profile["recent_chats"] = chats
    profile["updated_at"] = _now()

    uid = user.get("uid")

    if uid:

        try:

            db = _get_db()

            (
                db.collection("users")
                .document(uid)
                .set(
                    {
                        "recent_chats": chats,
                        "updated_at": profile[
                            "updated_at"
                        ],
                    },
                    merge=True,
                )
            )

        except Exception:
            pass

    return profile


# ============================================================
# EMAIL VALIDATION
# ============================================================

def _valid_email(
    email: str,
) -> bool:

    pattern = (
        r"^[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}$"
    )

    return bool(
        re.fullmatch(
            pattern,
            email,
        )
    )


# ============================================================
# SAVE CONTACT MESSAGE
# ============================================================

def save_contact_message(
    name: str,
    email: str,
    subject: str,
    message: str,
) -> bool:
    """
    Save a Contact Us message into Firestore.

    Collection:
        contact_messages

    This function intentionally raises a RuntimeError
    when Firestore fails so the application can show
    the actual technical reason.
    """

    # --------------------------------------------------------
    # CLEAN INPUT
    # --------------------------------------------------------

    name = str(name).strip()[:120]

    email = str(email).strip()[:254]

    subject = str(subject).strip()[:200]

    message = str(message).strip()[:5000]

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not name:

        raise ValueError(
            "Name is required."
        )

    if not email:

        raise ValueError(
            "Email is required."
        )

    if not _valid_email(email):

        raise ValueError(
            "Invalid email address."
        )

    if not message:

        raise ValueError(
            "Message is required."
        )

    # --------------------------------------------------------
    # CREATE FIRESTORE DOCUMENT
    # --------------------------------------------------------

    contact_data = {
        "name": name,
        "email": email,
        "subject": subject,
        "message": message,
        "created_at": _now(),
    }

    # --------------------------------------------------------
    # GET FIRESTORE
    # --------------------------------------------------------

    try:

        db = _get_db()

    except Exception as exc:

        raise RuntimeError(
            "Could not obtain Firestore client. "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    try:

        collection_ref = db.collection(
            "contact_messages"
        )

        document_ref = collection_ref.document()

        document_ref.set(
            contact_data
        )

        return True

    except Exception as exc:

        error_type = type(exc).__name__

        error_message = (
            str(exc).strip()
            or "No additional error information."
        )

        raise RuntimeError(
            "Firestore could not save the contact message. "
            f"{error_type}: {error_message}"
        ) from exc
