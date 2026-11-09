import os

from dotenv import load_dotenv


load_dotenv()


# =========================
# Groq Configuration
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# MySQL Configuration
# =========================

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")

MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

MYSQL_USER = os.getenv("MYSQL_USER", "root")

MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")

MYSQL_DATABASE = os.getenv(
    "MYSQL_DATABASE",
    "smartai_assistant",
)