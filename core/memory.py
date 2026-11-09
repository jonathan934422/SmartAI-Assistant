from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from core.database import get_connection


# ============================================================
# USER AUTHENTICATION
# ============================================================

def create_user(username, email, password):
    username = username.strip()
    email = email.strip()

    password_hash = generate_password_hash(password)

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users
                (username, email, password_hash)
            VALUES
                (%s, %s, %s)
            """,
            (username, email, password_hash),
        )

        connection.commit()

        return cursor.lastrowid

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_user_by_username(username):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT
                id,
                username,
                email,
                password_hash,
                created_at
            FROM users
            WHERE username = %s
            """,
            (username,),
        )

        return cursor.fetchone()

    finally:
        cursor.close()
        connection.close()


def authenticate_user(username, password):
    user = get_user_by_username(username)

    if not user:
        return None

    if not user.get("password_hash"):
        return None

    if check_password_hash(user["password_hash"], password):
        return user

    return None


def username_exists(username):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE username = %s
            """,
            (username,),
        )

        return cursor.fetchone() is not None

    finally:
        cursor.close()
        connection.close()


def email_exists(email):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = %s
            """,
            (email,),
        )

        return cursor.fetchone() is not None

    finally:
        cursor.close()
        connection.close()


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def _format_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()

    return str(value) if value else ""


def load_memory(user_id):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT
                id,
                title,
                created_at,
                updated_at
            FROM conversations
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )

        conversation_rows = cursor.fetchall()

        conversations = {}

        for row in conversation_rows:
            conversation_id = str(row["id"])

            conversations[conversation_id] = {
                "title": row["title"],
                "created_at": _format_datetime(
                    row["created_at"]
                ),
                "updated_at": _format_datetime(
                    row["updated_at"]
                ),
                "messages": [],
            }

        if not conversations:
            return conversations

        cursor.execute(
            """
            SELECT
                m.conversation_id,
                m.role,
                m.content,
                m.created_at
            FROM messages m
            INNER JOIN conversations c
                ON m.conversation_id = c.id
            WHERE c.user_id = %s
            ORDER BY m.id ASC
            """,
            (user_id,),
        )

        message_rows = cursor.fetchall()

        for row in message_rows:
            conversation_id = str(
                row["conversation_id"]
            )

            if conversation_id in conversations:
                conversations[conversation_id][
                    "messages"
                ].append(
                    {
                        "role": row["role"],
                        "content": row["content"],
                    }
                )

        return conversations

    finally:
        cursor.close()
        connection.close()


def save_memory(conversations):
    return True


def create_conversation(
    conversations,
    user_id,
    title,
    system_message,
):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO conversations
                (user_id, title)
            VALUES
                (%s, %s)
            """,
            (user_id, title),
        )

        conversation_id = cursor.lastrowid

        if system_message:
            cursor.execute(
                """
                INSERT INTO messages
                    (conversation_id, role, content)
                VALUES
                    (%s, %s, %s)
                """,
                (
                    conversation_id,
                    system_message.get(
                        "role",
                        "system",
                    ),
                    system_message.get(
                        "content",
                        "",
                    ),
                ),
            )

        connection.commit()

        conversation_id = str(conversation_id)

        conversations[conversation_id] = {
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": (
                [system_message.copy()]
                if system_message
                else []
            ),
        }

        return conversation_id

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_conversations(conversations):
    return sorted(
        conversations.items(),
        key=lambda item: item[1].get(
            "updated_at",
            "",
        ),
        reverse=True,
    )


def rename_conversation(
    conversations,
    conversation_id,
    title,
    user_id,
):
    title = title.strip()

    if not title:
        return

    if conversation_id not in conversations:
        return

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            UPDATE conversations
            SET title = %s
            WHERE id = %s
              AND user_id = %s
            """,
            (
                title,
                int(conversation_id),
                user_id,
            ),
        )

        connection.commit()

        conversations[conversation_id]["title"] = title
        conversations[conversation_id][
            "updated_at"
        ] = datetime.now().isoformat()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def delete_conversation(
    conversations,
    conversation_id,
    user_id,
):
    if conversation_id not in conversations:
        return

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            DELETE FROM conversations
            WHERE id = %s
              AND user_id = %s
            """,
            (
                int(conversation_id),
                user_id,
            ),
        )

        connection.commit()

        conversations.pop(
            conversation_id,
            None,
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


# ============================================================
# SECURE MESSAGE STORAGE
# ============================================================

def save_message(
    conversation_id,
    user_id,
    role,
    content,
):
    """
    Save a message only if the conversation belongs
    to the currently logged-in user.
    """

    connection = get_connection()
    cursor = connection.cursor()

    try:
        # ----------------------------------------------------
        # SECURITY CHECK
        # ----------------------------------------------------
        cursor.execute(
            """
            SELECT id
            FROM conversations
            WHERE id = %s
              AND user_id = %s
            """,
            (
                int(conversation_id),
                user_id,
            ),
        )

        conversation = cursor.fetchone()

        if not conversation:
            raise PermissionError(
                "You do not have permission to access "
                "this conversation."
            )

        # ----------------------------------------------------
        # SAVE MESSAGE
        # ----------------------------------------------------
        cursor.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content)
            VALUES
                (%s, %s, %s)
            """,
            (
                int(conversation_id),
                role,
                content,
            ),
        )

        # ----------------------------------------------------
        # UPDATE CONVERSATION TIME
        # ----------------------------------------------------
        cursor.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
              AND user_id = %s
            """,
            (
                int(conversation_id),
                user_id,
            ),
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()