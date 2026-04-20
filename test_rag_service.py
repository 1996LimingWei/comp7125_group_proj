import unittest
import types
import sys
from unittest.mock import patch


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


class _FakeClient:
    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, *, name, metadata):
        return self._collection

    def delete_collection(self, *, name):
        return None


class TestRAGService(unittest.TestCase):
    def test_vector_store_filters_non_chunk_results(self):
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
                "chromadb": fake_chromadb,
                "chromadb.config": fake_chromadb_config,
            },
        ):
            from src.rag.vector_store import ChromaVectorStore

            store = ChromaVectorStore(chroma_path="./chroma_db")
            chunks = store.query(query_embedding=[0.0, 0.0], top_k=5)

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].source, "A.txt")
            self.assertEqual(chunks[0].chunk_id, 1)

    def test_neural_search_is_exposed(self):
        fake_collection = _FakeCollection(count_value=0, query_result={})
        fake_client = _FakeClient(fake_collection)

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
                "chromadb": fake_chromadb,
                "chromadb.config": fake_chromadb_config,
            },
        ):
            from src.rag.neural_search import neural_search

            self.assertTrue(callable(neural_search))

    def test_answer_with_neural_rag_is_exposed_and_uses_citations(self):
        from src.rag.types import RetrievedChunk

        with patch("src.rag.neural_search.neural_search") as fake_search, patch(
            "src.rag.neural_search.ollama_generate"
        ) as fake_generate:
            fake_search.return_value = [
                RetrievedChunk(
                    content="chunk text",
                    source="A.txt",
                    chunk_id=1,
                    start_token=0,
                    end_token=10,
                    distance=0.25,
                )
            ]
            fake_generate.return_value = "answer [1]"

            from src.rag.neural_search import answer_with_neural_rag

            out = answer_with_neural_rag("test query", top_k=1)
            self.assertEqual(out, "answer [1]")

            called_prompt = fake_generate.call_args.kwargs.get("prompt") or fake_generate.call_args.args[0]
            self.assertIn("Cite snippet numbers like [1][2].", called_prompt)
            self.assertIn("[Snippet 1]", called_prompt)
            self.assertIn("Source: A.txt#chunk:1", called_prompt)

    def test_snippet_shape_is_module4_compatible(self):
        from src.rag.types import RetrievedChunk, retrieved_chunks_to_snippets

        retrieved = [
            RetrievedChunk(
                content="hello",
                source="A.txt",
                chunk_id=7,
                start_token=0,
                end_token=10,
                distance=0.25,
            )
        ]
        snippets = retrieved_chunks_to_snippets(retrieved, citation_prefix="N", retriever="neural")
        self.assertEqual(len(snippets), 1)
        self.assertEqual(snippets[0]["citation_key"], "N0")
        self.assertEqual(snippets[0]["text"], "hello")
        self.assertEqual(snippets[0]["meta"]["file_name"], "A.txt")
        self.assertEqual(snippets[0]["meta"]["chunk_id"], 7)
        self.assertIn("score", snippets[0]["meta"])
        self.assertIn("distance", snippets[0]["meta"])
        self.assertEqual(snippets[0]["meta"]["retriever"], "neural")

    def test_course_code_query_is_reranked(self):
        fake_collection = _FakeCollection(count_value=0, query_result={})
        fake_client = _FakeClient(fake_collection)

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
                "chromadb": fake_chromadb,
                "chromadb.config": fake_chromadb_config,
            },
        ):
            from src.rag.service import RAGService
            from src.rag.types import RetrievedChunk

            class _FakeEmbedder:
                embedding_id = "fake"

                def embed_query(self, text):
                    return [0.0, 0.0]

            candidates = [
                RetrievedChunk(
                    content="General student life info...",
                    source="Student_Life.txt",
                    chunk_id=0,
                    start_token=None,
                    end_token=None,
                    distance=0.20,
                ),
                RetrievedChunk(
                    content="Prerequisite:\nCOMP7015 Artificial Intelligence or COMP7025 Artificial Intelligence for Digital Transformation",
                    source="COMP7125_Course_Outline.txt",
                    chunk_id=0,
                    start_token=None,
                    end_token=None,
                    distance=0.41,
                ),
                RetrievedChunk(
                    content="Certification of academic assessment info...",
                    source="Certification_of_Academic_Assessment.txt",
                    chunk_id=0,
                    start_token=None,
                    end_token=None,
                    distance=0.21,
                ),
            ]

            class _FakeStore:
                def count(self):
                    return len(candidates)

                def query(self, *, query_embedding, top_k):
                    return candidates[:top_k]

            rag = RAGService(
                rebuild_if_changed=False,
                embedder=_FakeEmbedder(),
                vector_store=_FakeStore(),
            )
            got = rag.retrieve_chunks("What are the prerequisites for COMP7125?", k=2)
            self.assertEqual(got[0].source, "COMP7125_Course_Outline.txt")


if __name__ == "__main__":
    unittest.main()

