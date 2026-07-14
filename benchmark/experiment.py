import os
import pandas as pd
from benchmark.evaluation import evaluate_pipeline


PDF_PATH = "data/pdfs/Telephonic uncanny.pdf"
BENCHMARK_PATH = "benchmark/benchmark.csv"


embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2"
]

chunk_sizes = [300, 500, 700, 900, 1200]
overlaps = [0, 50, 100, 150]

experiment_results = []

print("\nStarting Full RAG Experiments...\n")


for model_name in embedding_models:

    print("\n===================================")
    print(f"Embedding Model: {model_name}")
    print("===================================\n")

    for chunk_size in chunk_sizes:
        for overlap in overlaps:

            print(f"Running: model={model_name}, chunk={chunk_size}, overlap={overlap}")

            metrics, _ = evaluate_pipeline(
                pdf_path="data/Telephonic uncanny.pdf",
                benchmark_path="benchmark/benchmark.csv",
                chunk_size=300,
                overlap=100,
                embedding_model="BAAI/bge-small-en-v1.5",
                retrieval_method="hybrid" # IMPORTANT CHANGE
            )

            experiment_results.append({
                "Embedding Model": model_name,
                "Chunk Size": chunk_size,
                "Overlap": overlap,
                "Recall@1": round(metrics["Recall@1"], 2),
                "Recall@3": round(metrics["Recall@3"], 2),
                "Recall@5": round(metrics["Recall@5"], 2)
            })


results_df = pd.DataFrame(experiment_results)

os.makedirs("benchmark/results", exist_ok=True)

output_path = "benchmark/results/full_experiments.csv"

results_df.to_csv(output_path, index=False)

print("\n==============================")
print(" ALL EXPERIMENTS COMPLETE ")
print("==============================")

print(results_df.sort_values("Recall@1", ascending=False))

print(f"\nSaved to:\n{output_path}")