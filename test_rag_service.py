import unittest
import types
import sys
from unittest.mock import patch


class _FakeArray:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        return _FakeArray([[0.0, 0.0] for _ in texts])


class _FakeCollection:
    def __init__(self, *, count_value=1, query_result=None):
        self._count_value = count_value
        self._query_result = query_result or {}

    def count(self):
        return self._count_value

    def query(self, *, query_embeddings, n_results, include, where):
        if where != {"doc_type": "chunk"}:
            raise AssertionError(f"Expected where={{'doc_type':'chunk'}}, got {where}")
        return self._query_result

    def add(self, *, documents, ids, metadatas, embeddings):
        return None

    def get(self, *, ids, include):
        return {"documents": []}


class _FakeClient:
    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, *, name, metadata):
        return self._collection

    def delete_collection(self, *, name):
        return None


class TestRAGService(unittest.TestCase):
    def test_retrieve_chunks_filters_manifest_defensively(self):
        query_result = {
            "documents": [["chunk text", "manifest text"]],
            "metadatas": [[
                {"doc_type": "chunk", "file_name": "A.txt", "chunk_id": 1},
                {"doc_type": "manifest"},
            ]],
            "distances": [[0.1, 0.0]],
        }
        fake_collection = _FakeCollection(count_value=2, query_result=query_result)
        fake_client = _FakeClient(fake_collection)

        fake_sentence_transformers = types.ModuleType("sentence_transformers")
        fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer

        fake_chromadb = types.ModuleType("chromadb")
        fake_chromadb.PersistentClient = lambda *args, **kwargs: fake_client

        fake_chromadb_config = types.ModuleType("chromadb.config")

        class _FakeSettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_chromadb_config.Settings = _FakeSettings

        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": fake_sentence_transformers,
                "chromadb": fake_chromadb,
                "chromadb.config": fake_chromadb_config,
            },
        ):
            from src.rag.service import RAGService

            rag = RAGService(
                data_dir="./course_docs",
                chroma_path="./chroma_db",
                rebuild_if_changed=False,
            )
            chunks = rag.retrieve_chunks("test query", k=5)

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].source, "A.txt")
            self.assertEqual(chunks[0].chunk_id, 1)


if __name__ == "__main__":
    unittest.main()

