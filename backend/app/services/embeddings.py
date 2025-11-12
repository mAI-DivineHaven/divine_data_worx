"""
DivineHaven Embeddings Service
Handles query embedding generation via Ollama.
"""

import asyncio

import httpx


class EmbeddingsService:
    """Service for generating embeddings via Ollama API."""

    def __init__(
        self,
        api_base: str = "http://localhost:11434",
        model: str = "embeddinggemma",
        timeout: float = 30.0,
        max_concurrency: int = 8,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        concurrency = max(1, max_concurrency)
        self._semaphore = asyncio.Semaphore(concurrency)

    async def embed_async(self, text: str) -> list[float]:
        """
        Generate embedding for a single text asynchronously.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            httpx.HTTPError: If Ollama API request fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns {"embeddings": [[...]]}
            if "embeddings" in data and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            elif "embedding" in data:  # Fallback for older API
                return data["embedding"]
            else:
                raise ValueError(f"Unexpected Ollama response format: {data.keys()}")

    def embed_sync(self, text: str) -> list[float]:
        """
        Generate embedding for a single text synchronously.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            httpx.HTTPError: If Ollama API request fails
        """
        with httpx.Client() as client:
            response = client.post(
                f"{self.api_base}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if "embeddings" in data and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            elif "embedding" in data:
                return data["embedding"]
            else:
                raise ValueError(f"Unexpected Ollama response format: {data.keys()}")

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per text)
        """

        async def _embed(text: str) -> list[float]:
            async with self._semaphore:
                return await self.embed_async(text)

        tasks = [asyncio.create_task(_embed(text)) for text in texts]
        return await asyncio.gather(*tasks)

    def check_health(self) -> bool:
        """
        Check if Ollama is reachable and model is available.

        Returns:
            True if healthy, False otherwise
        """
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.api_base}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return self.model in models or any(self.model in m for m in models)
                return False
        except Exception:
            return False
