import psycopg2
from .config import PG_DB,PG_USER, PG_PASSWORD, PG_HOST, PG_PORT

def get_conn():
    return psycopg2.connect(
        dbname = PG_DB, user = PG_USER, password = PG_PASSWORD,
        host = PG_HOST, port = PG_PORT
    )