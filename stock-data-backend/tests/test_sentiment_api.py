import pytest
from datetime import datetime


def test_sentiment_backfill_endpoint(client, db_storage):
    """
    Tests the sentiment analysis backfill endpoint.
    """
    # 1. Insert a dummy news article
    with db_storage.pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO news_sentiment (published_at, stock_symbol, headline, status)
                   VALUES (%s, %s, %s, %s) RETURNING id;""",
                (datetime.utcnow(), 'DUMMY', 'This is a test headline.', 'PENDING')
            )
            article_id = cursor.fetchone()[0]

    # 2. Call the backfill endpoint
    response = client.post("/sentiment/backfill")
    assert response.status_code == 200
    assert "Sentiment analysis backfill completed" in response.json["message"]

    # 3. Verify the article was processed
    with db_storage.pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT status, sentiment_score FROM news_sentiment WHERE id = %s;",
                (article_id,)
            )
            status, score = cursor.fetchone()
            assert status == 'PROCESSED'
            assert score is not None
