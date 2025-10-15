import pytest
from datetime import datetime
from concurrent.futures import Executor, Future, as_completed

# A mock executor that runs tasks serially in the same process for testing
class SerialExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._shutdown = False

    def submit(self, fn, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')

        future = Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            future.set_exception(e)
        else:
            future.set_result(result)
        
        return future

    def shutdown(self, wait=True):
        self._shutdown = True

# A mock storage class to completely isolate the test from the database
class MockStorage:
    def get_all_pending_sentiment_articles(self):
        return [(1, 'dummy headline', 'dummy content')]

    def update_sentiment_scores(self, results):
        # In a real scenario, you might check the content of 'results'
        pass

    def close(self):
        pass

def test_sentiment_backfill_endpoint(client, monkeypatch):
    """
    Tests the sentiment analysis backfill endpoint in complete isolation from the database.
    """
    # --- Isolate the endpoint from ALL external systems ---

    # 1. Use a completely fake storage object to avoid any real DB interaction
    mock_storage_instance = MockStorage()
    monkeypatch.setattr('routes.sentiment.Storage', lambda config: mock_storage_instance)

    # 2. Prevent endpoint from creating its own Config object (file I/O)
    monkeypatch.setattr('routes.sentiment.Config', lambda: None)

    # 3. Mock concurrency and ML model loading
    monkeypatch.setattr('routes.sentiment.ProcessPoolExecutor', SerialExecutor)
    monkeypatch.setattr('routes.sentiment.analyze_sentiment', lambda title, content: 0.75)
    monkeypatch.setattr('routes.sentiment.as_completed', lambda futures: futures)

    # --- Test Logic ---

    # Call the backfill endpoint
    response = client.post("/sentiment/backfill")
    assert response.status_code == 200
    assert "Sentiment analysis backfill completed" in response.json["message"]
