from rag_pipeline.evaluation import EvalSample, run_eval_suite


# ===== DEFINE YOUR GROUND TRUTH =====
# You have manual + telemetry sources.
# Define ground truth like this:

eval_samples = [
    EvalSample(
        query="motor overheating during climb",
        expected_ids=[
            "flight01_normal.csv_telemetry", 
            "uav_manual.pdf_manual"
        ],
        description="ESC temp and motor overload"
    ),

    EvalSample(
        query="gps lost after high vibration",
        expected_ids=[
            "biglog_static_imu.csv_telemetry",
        ],
        description="IMU vibration → GPS dropout"
    ),
]

# ===== RUN EVAL =====
k = 5
results = run_eval_suite(eval_samples, k=k)

print("\n=== Evaluation Summary ===")
print(f"Precision@{k}: {results['precision@k']:.3f}")
print(f"Recall@{k}:    {results['recall@k']:.3f}")
print(f"MRR:           {results['MRR']:.3f}")

print("\n=== Per-sample details ===")
for idx, r in enumerate(results["samples"], start=1):
    print(f"\nSample {idx}:")
    print(f"Hits: {r.hits}/{r.total_expected}")
    print(f"Precision@k: {r.precision_at_k:.3f}")
    print(f"Recall@k:    {r.recall_at_k:.3f}")
    print(f"MRR:         {r.mrr:.3f}")

print("\nDone.")
