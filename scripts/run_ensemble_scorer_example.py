"""
This script demonstrates how to run the MLCache system with an EnsembleScorer.

It covers the following steps:
1.  **Setting up the Environment**: Importing necessary modules and adding the source
    directory to the system path.
2.  **Simulating a Data Stream**: Creating mock data and a stream adapter to simulate
    a real-time flow of queries. In a real-world scenario, this adapter would
    connect to a live data source like Kafka, a log file, or a network stream.
3.  **Configuring the Ensemble Scorer**: Building an EnsembleScorer, which combines
    the outputs of multiple individual scorers to make a more robust decision.
4.  **Building the MLCache Runtime**: Instantiating the main MLCache runtime, which
    orchestrates the caching logic, using the configured EnsembleScorer.
5.  **Indexing Cache Candidates**: Populating the cache with some initial data that
    can be retrieved.
6.  **Processing the Stream**: Running the main loop to process incoming queries from
    the stream, look them up in the cache, and print the decisions.
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np

# --- 1. Environment Setup ---
# Add the project's source directory to the Python path to allow direct imports
# of the library modules.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the necessary components from the MLCache library.
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.runtime import (
    MLCacheRuntimeConfig,
    build_local_mlcache_runtime,
)
from mlcache.scorers import CosineScorer, EnsembleScorer, LDAScorer
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, Embedding, Query, Response, OracleDecision
from mlcache.feedback import JudgeRequest, JudgeResponse, JudgeLabel, Judge


# --- 2. Data Stream Simulation ---
# In a real application, you would have a stream of incoming queries. Here, we
# simulate this by creating a mock data source and a simple adapter.

# Define a simple structure for our mock data records.
class MockRecord(NamedTuple):
    query_id: str
    query_text: str
    embedding: np.ndarray
    is_positive_example: bool  # True if this is an H1 example (should hit), False for H0

# Generate some mock data for our stream.
# We create two clusters of embeddings: one for "positive" examples that should
# ideally hit in the cache, and one for "negative" examples that should miss.
DIMENSION = 64
CENTER_A = np.ones(DIMENSION)
CENTER_B = -np.ones(DIMENSION)

MOCK_STREAM_DATA: list[MockRecord] = [
    # Positive examples (close to CENTER_A)
    MockRecord("q1", "What is the capital of France?", CENTER_A + np.random.randn(DIMENSION) * 0.1, True),
    MockRecord("q2", "Best restaurants in Paris?", CENTER_A + np.random.randn(DIMENSION) * 0.1, True),
    # Negative examples (close to CENTER_B)
    MockRecord("q3", "How to bake a cake?", CENTER_B + np.random.randn(DIMENSION) * 0.1, False),
    MockRecord("q4", "Latest news in tech?", CENTER_B + np.random.randn(DIMENSION) * 0.1, False),
]

# This adapter turns our list of mock data into a "stream" of CacheLookups.
class MockStreamAdapter:
    """
    A simple adapter that yields CacheLookup objects from a list of mock records.
    In a real system, this would read from a message queue or another data source.
    """
    def __init__(self, records: list[MockRecord]):
        self._records = records

    def __iter__(self) -> Iterable[tuple[MockRecord, CacheLookup]]:
        for record in self._records:
            lookup = CacheLookup(
                query=Query(record.query_text),
                embedding=record.embedding,
            )
            yield record, lookup

# A mock judge that provides ground truth labels for our simulated data.
# The runtime can use this to evaluate the quality of its decisions.
class MockJudge(Judge):
    def judge(self, request: JudgeRequest) -> JudgeResponse:
        # For this example, we simply say a candidate is reusable if the incoming
        # query was marked as a positive example. This is a simplification.
        is_positive = request.context.get("is_positive_example", False)
        label = JudgeLabel.REUSABLE if is_positive else JudgeLabel.NOT_REUSABLE
        return JudgeResponse(request_id=request.request_id, decision=label)

# --- 3. Ensemble Scorer Configuration ---
# The EnsembleScorer combines multiple scorers. It's useful when different
# scorers capture different aspects of similarity.
def build_example_ensemble_scorer() -> EnsembleScorer:
    """
    Creates an EnsembleScorer composed of a CosineScorer and an LDAScorer.
    The EnsembleScorer will run its constituent scorers (judges) and aggregate
    their scores to make a final decision.
    """
    print("Building the Ensemble Scorer with Cosine and LDA scorers...")
    return EnsembleScorer(
        judges=[
            CosineScorer(),
            LDAScorer(),
        ]
    )


# --- 4. MLCache Runtime Construction ---
def setup_runtime() -> Any:
    """
    Initializes and configures the MLCache runtime.
    """
    print("Setting up the MLCache runtime...")
    # The runtime configuration allows you to control various behaviors.
    # Here, we use a simple default configuration.
    runtime_config = MLCacheRuntimeConfig()

    # The FeatureBuilder preprocesses embeddings before they are scored.
    feature_builder = NormalizedHadamardFeatureBuilder()

    # Instantiate our ensemble scorer.
    ensemble_scorer = build_example_ensemble_scorer()
    
    # A mock judge for evaluation purposes
    mock_judge = MockJudge()

    # `build_local_mlcache_runtime` is a factory function that assembles all the
    # components into a ready-to-use runtime instance.
    # We set `use_file_persistence=False` to use in-memory stores, which is
    # convenient for this example.
    runtime = build_local_mlcache_runtime(
        root_dir="./ensemble_runtime_state",
        feature_builder=feature_builder,
        scorer=ensemble_scorer,
        judge=mock_judge,
        config=runtime_config,
        use_file_persistence=False,
    )
    print("Runtime setup complete.")
    return runtime


# --- 5. Indexing Cache Candidates ---
def index_initial_data(runtime: Any):
    """
    Populates the runtime's vector store with some initial cache entries.
    These entries are the candidates that the runtime will try to match against.
    """
    print("Indexing initial cache entries...")
    # We will create a cache entry that is similar to our "positive" examples.
    initial_entry = CacheEntry(
        cache_key=CacheKey(str(uuid.uuid4())),
        query=Query("Capital of France"),
        response=Response("The capital of France is Paris."),
        embedding=CENTER_A,  # This embedding is close to our positive examples
    )
    runtime.put(initial_entry)
    print(f"Indexed entry with key: {initial_entry.cache_key}")


# --- 6. Processing the Stream ---
def process_stream(runtime: Any, stream_adapter: MockStreamAdapter):
    """
    Iterates through the data stream and processes each query with the runtime.
    """
    print("
--- Starting Stream Processing ---")
    for record, lookup in stream_adapter:
        print(f"
Processing query: '{lookup.query}' (positive_example: {record.is_positive_example})")

        # The core of the application: perform a lookup in the cache.
        # The runtime uses the configured ensemble scorer to find the best match.
        result: OracleDecision = runtime.lookup_with_decision(lookup, context={"is_positive_example": record.is_positive_example}).decision

        # Print the result of the lookup.
        if result.accepted:
            print(f"  -> Cache HIT!")
            print(f"     Accepted Key: {result.cache_key}")
            print(f"     Score: {result.score:.4f} (Threshold: {result.threshold})")
        else:
            print(f"  -> Cache MISS.")
            if result.reason:
                print(f"     Reason: {result.reason}")

    print("
--- Stream Processing Finished ---
")


def main():
    """
    Main function to orchestrate the example.
    """
    # Initialize the runtime with our ensemble scorer configuration.
    runtime = setup_runtime()

    # Add some data to the cache that can be looked up.
    index_initial_data(runtime)

    # Create an adapter for our simulated stream of data.
    stream_adapter = MockStreamAdapter(MOCK_STREAM_DATA)

    # Process the stream and print the results.
    process_stream(runtime, stream_adapter)


if __name__ == "__main__":
    main()

