import os

from benchmark.evaluation import evaluate_pipeline


def main():

    metrics, results_df = evaluate_pipeline(
        pdf_path="data/pdfs/Telephonic uncanny.pdf",
        benchmark_path="benchmark/benchmark.csv",
        chunk_size=500,
        overlap=100
    )

    print("\n==============================")
    print(" Retrieval Evaluation Results ")
    print("==============================")

    print(f"Recall@1 : {metrics['Recall@1']:.2f}%")
    print(f"Recall@3 : {metrics['Recall@3']:.2f}%")
    print(f"Recall@5 : {metrics['Recall@5']:.2f}%")

    os.makedirs("benchmark/results", exist_ok=True)

    output_path = "benchmark/results/evaluation_results.csv"

    results_df.to_csv(output_path, index=False)

    print(f"\nDetailed results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()