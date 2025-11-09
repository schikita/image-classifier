from .connection import get_conn

class ImageRepo:
    def insert_pending(self, *, name, path, height, width,format, size_mb, file_hash):
        sql = """
        INSERT INTO image (name, path, height, width, format, size_mb, file_hash, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
        ON CONFLICT(file_hash) DO NOTHING
        RETURNING ID;
        """
        with get_conn() as c, c.cursor() as cur:
            cur.execute(sql,(name, path, height, width, format, size_mb, file_hash ))
            row = cur.fetchone()
            return row[0] if row else None
    
    def set_prediction(self, image_id, label, confidence):
        sql = """
        UPDATE image
        SET predicted_label = %s,
            confidence = %s,
            status = 'auto_labeled'
        WHERE id = %s;
        """
        with get_conn() as c, c.cursor() as cur:
            cur.execute(sql, (label, float(confidence), image_id))

    def confirm(self, image_id, true_label):
        with get_conn() as c, c.cursor() as cur:
            cur.execute (
                "UPDATE image SET true_label = %s, status = 'confimed' WHERE id =  %s RETURNING id;",
                (true_label, image_id)
            )
            return cur.fetchone() is not None

