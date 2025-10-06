import json

def test_health_check(client):
    """
    Tests the /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "healthy"}

def test_backfill_features(client):
    """
    Tests the /features/backfill endpoint.
    """
    response = client.post(
        "/features/backfill",
        data=json.dumps({"start_date": "2023-01-01", "end_date": "2023-01-31"}),
        content_type="application/json",
    )
    assert response.status_code == 202
    json_data = response.get_json()
    assert "Successfully started feature backfill" in json_data["message"]

def test_full_backfill_features(client):
    """
    Tests the /features/full-backfill endpoint.
    """
    response = client.post("/features/full-backfill")
    assert response.status_code == 202
    json_data = response.get_json()
    assert "Successfully started parallel full feature backfill" in json_data["message"]
