import io
import os
import pytest
from fastapi.testclient import TestClient

# ensure env for dummy path
os.environ.pop("WHISPER_ENDPOINT_URL", None)
os.environ.pop("RAG_ENDPOINT_URL", None)

from serving.app_main import app

client = TestClient(app)

def test_missing_rag_url():
    # uploads dummy .wav → should 500 on missing RAG_URL
    audio = io.BytesIO(b"")
    resp = client.post("/predict", files={"file": ("a.wav", audio, "audio/wav")})
    assert resp.status_code == 500
    assert "RAG_ENDPOINT_URL not set" in resp.json()["detail"]

@pytest.mark.parametrize("whisper_url,rag_url,expected_query", [
    (None, "http://fake-rag/predict", "[DUMMY] a.wav"),
])
def test_dummy_transcribe_and_rag(monkeypatch, whisper_url, rag_url, expected_query):
    if whisper_url:
        monkeypatch.setenv("WHISPER_ENDPOINT_URL", whisper_url)
    monkeypatch.setenv("RAG_ENDPOINT_URL", rag_url)

    # monkeypatch the RAG POST
    def fake_post(url, json=None, files=None):
        class R: 
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"answer": "OK", "query": json["query"]}
        return R()
    monkeypatch.setattr("requests.post", fake_post)

    audio = io.BytesIO(b"")
    resp = client.post("/predict", files={"file": ("a.wav", audio, "audio/wav")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == expected_query
    assert data["answer"]["answer"] == "OK"
