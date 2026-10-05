from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from firebase_service import _db
except Exception:
    _db = lambda: None


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
# TIME HELPERS
# ============================================================

def _now() -> str:
    """
    Return current UTC timestamp as an ISO-8601 string.
    """
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# PROFILE DEFAULTS
# ============================================================

def _defaults(user: dict[str, Any]) -> dict[str, Any]:
    """
    Create the default AuraAI profile structure.
    """

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
# PROGRESS CALCULATION
# ============================================================

def _calculate_progress(
    completed: list[str],
) -> int:
    """
    Calculate profile completion progress.

    Unknown completion items are ignored.
    Duplicate items are counted only once.
    """

    return min(
        100,
        sum(
            PROGRESS_KEYS.get(item, 0)
            for item in set(completed)
        ),
    )


# ============================================================
# FIRESTORE DATABASE HELPER
# ============================================================

def _get_database():
    """
    Safely obtain the Firebase Admin Firestore client.

    Returns:
        Firestore client or None.
    """

    try:
        db = _db()
        return db

    except Exception:
        return None


# ============================================================
# GET USER PROFILE
# ============================================================

def get_profile(
    user: dict[str, Any],
) -> dict[str, Any]:
    """
    Retrieve a user's profile from Firestore.

    If the profile does not exist, default values are returned.
    """

    profile = _defaults(user)

    db = _get_database()

    uid = user.get("uid")

    if db is not None and uid:

        try:

            document = (
                db.collection("users")
                .document(uid)
                .get()
            )

            data = (
                document.to_dict()
                if document.exists
                else {}
            )

            if data:
                profile.update(data)

        except Exception:
            # Profile retrieval should never crash the UI.
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
# MARK PROFILE ITEM COMPLETE
# ============================================================

def mark_complete(
    user: dict[str, Any],
    item: str,
) -> dict[str, Any]:
    """
    Mark a particular AuraAI section as completed.

    Example:
        mark_complete(user, "chat")
    """

    item = str(item).strip()

    if not item:
        return get_profile(user)

    profile = get_profile(user)

    completed = set(
        profile.get("completed") or []
    )

    completed.add(item)

    profile["completed"] = sorted(
        completed
    )

    profile["progress"] = _calculate_progress(
        profile["completed"]
    )

    profile["updated_at"] = _now()

    db = _get_database()

    uid = user.get("uid")

    if db is not None and uid:

        try:

            (
                db.collection("users")
                .document(uid)
                .set(
                    {
                        "completed": profile[
                            "completed"
                        ],
                        "progress": profile[
                            "progress"
                        ],
                        "updated_at": profile[
                            "updated_at"
                        ],
                    },
                    merge=True,
                )
            )

        except Exception:
            # Progress persistence should not crash AuraAI.
            pass

    return profile


# ============================================================
# SET CAREER GOAL
# ============================================================

def set_goal(
    user: dict[str, Any],
    goal: str,
) -> dict[str, Any]:
    """
    Save the user's career goal and mark the goal section
    as completed.
    """

    goal = str(goal).strip()[:500]

    profile = get_profile(user)

    profile["career_goal"] = goal
    profile["updated_at"] = _now()

    db = _get_database()

    uid = user.get("uid")

    if db is not None and uid:

        try:

            (
                db.collection("users")
                .document(uid)
                .set(
                    {
                        "career_goal": goal,
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
    """
    Store the user's latest chat prompts.

    Maximum:
        8 recent conversations.
    """

    profile = get_profile(user)

    item = {
        "title": str(title).strip()[:120],
        "prompt": str(prompt).strip()[:1000],
        "timestamp": _now(),
    }

    chats = [
        item,
        *list(
            profile.get("recent_chats") or []
        ),
    ]

    profile["recent_chats"] = chats[:8]
    profile["updated_at"] = _now()

    db = _get_database()

    uid = user.get("uid")

    if db is not None and uid:

        try:

            (
                db.collection("users")
                .document(uid)
                .set(
                    {
                        "recent_chats": profile[
                            "recent_chats"
                        ],
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
# SAVE CONTACT MESSAGE
# ============================================================

def save_contact_message(
    name: str,
    email: str,
    subject: str,
    message: str,
) -> bool:
    """
    Save a Contact Us submission to Firestore.

    Firestore collection:
        contact_messages

    Every submission receives a new automatically generated
    Firestore document ID.

    Raises:
        RuntimeError:
            When Firestore is unavailable or the write fails.

    Returns:
        True when the message is successfully saved.
    """

    # --------------------------------------------------------
    # INPUT NORMALIZATION
    # --------------------------------------------------------

    clean_name = (
        str(name)
        .strip()
        [:120]
    )

    clean_email = (
        str(email)
        .strip()
        [:254]
    )

    clean_subject = (
        str(subject)
        .strip()
        [:200]
    )

    clean_message = (
        str(message)
        .strip()
        [:5000]
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if not clean_name:

        raise ValueError(
            "Contact name cannot be empty."
        )

    if not clean_email:

        raise ValueError(
            "Contact email cannot be empty."
        )

    if not clean_message:

        raise ValueError(
            "Contact message cannot be empty."
        )

    # --------------------------------------------------------
    # EMAIL VALIDATION
    # --------------------------------------------------------

    if (
        "@" not in clean_email
        or "." not in clean_email.split("@")[-1]
    ):

        raise ValueError(
            "Please enter a valid email address."
        )

    # --------------------------------------------------------
    # FIRESTORE DATA
    # --------------------------------------------------------

    values = {
        "name": clean_name,
        "email": clean_email,
        "subject": clean_subject,
        "message": clean_message,
        "created_at": _now(),
    }

    # --------------------------------------------------------
    # GET FIRESTORE CLIENT
    # --------------------------------------------------------

    try:

        db = _get_database()

    except Exception as exc:

        raise RuntimeError(
            "Unable to initialize the Firestore client: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if db is None:

        raise RuntimeError(
            "Firestore Admin SDK is not initialized. "
            "Check the [firebase] section in Streamlit "
            "Secrets and make sure the Firebase Admin "
            "service-account configuration is valid."
        )

    # --------------------------------------------------------
    # WRITE CONTACT MESSAGE
    # --------------------------------------------------------

    try:

        collection = db.collection(
            "contact_messages"
        )

        result = collection.add(
            values
        )

        # Firebase Admin SDK normally returns:
        # (DocumentReference, WriteResult)
        #
        # We intentionally do not expose document IDs
        # to the user.

        if result is None:

            raise RuntimeError(
                "Firestore returned no result after "
                "the contact message write."
            )

        return True

    except Exception as exc:

        error_type = type(exc).__name__
        error_message = str(exc).strip()

        if not error_message:
            error_message = (
                "No additional Firebase error message "
                "was returned."
            )

        raise RuntimeError(
            "Firestore contact_messages write failed. "
            f"{error_type}: {error_message}"
        ) from exc
