import mysql.connector

from config.settings import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
)


def get_connection():
    """Create and return a MySQL database connection."""

    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )


def test_connection():
    """Test the SmartAI MySQL database connection."""

    connection = None

    try:
        connection = get_connection()

        if connection.is_connected():
            return True

        return False

    except mysql.connector.Error as error:
        print(f"MySQL connection error: {error}")
        return False

    finally:
        if connection and connection.is_connected():
            connection.close()