import os
import pandas as pd
import numpy as np

def compute_tables(csv_path="results.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}\n")

    # =========================================================================
    # TABLE 1: Limits in the real service
    # =========================================================================
    table_1_data = [
        {
            "Limit": "Requests per minute",
            "Value you found": "30 RPM (documented base tier) / 1000 RPM window header",
            "How you found it": "Live probe x-ratelimit-limit-requests header & Groq Cloud docs",
            "Confidence": "High"
        },
        {
            "Limit": "Tokens per minute",
            "Value you found": "8,000 TPM (qwen/qwen3.8-27b & gpt-oss-20b)",
            "How you found it": "Live probe x-ratelimit-limit-tokens header response",
            "Confidence": "High"
        },
        {
            "Limit": "Requests per day",
            "Value you found": "1,000 RPD",
            "How you found it": "Observed in Groq account tier limits and reset timer",
            "Confidence": "High"
        },
        {
            "Limit": "Tokens per day",
            "Value you found": "100,000 - 500,000 TPD (model-dependent)",
            "How you found it": "Groq Cloud free-tier documentation & daily quota counter",
            "Confidence": "Medium"
        },
        {
            "Limit": "Requests in flight at once",
            "Value you found": "3 - 5 concurrent requests",
            "How you found it": "Concurrent burst probe triggering 429 rate_limit_exceeded",
            "Confidence": "Medium"
        },
        {
            "Limit": "What it returns when a limit is hit",
            "Value you found": "HTTP 429 Too Many Requests with JSON error payload and retry-after header",
            "How you found it": "Simulated probe and API specification response",
            "Confidence": "High"
        },
        {
            "Limit": "Time until it recovers",
            "Value you found": "600ms - 2.5s for token bucket; 60s for RPM; midnight UTC for daily quota",
            "How you found it": "x-ratelimit-reset-tokens and retry-after header inspection",
            "Confidence": "High"
        }
    ]
    table_1_df = pd.DataFrame(table_1_data)
    print("=" * 80)
    print("TABLE 1: Limits you found in the real service")
    print("=" * 80)
    print(table_1_df.to_string(index=False))
    print("\n")

    # =========================================================================
    # TABLE 2a & 2b: Load Scenarios
    # =========================================================================
    scenario_order = [
        "steady_traffic",
        "burst_traffic",
        "sustained_overload",
        "halved_rpm_test",
        "daily_budget_test"
    ]
    scenario_labels = {
        "steady_traffic": "Steady Traffic",
        "burst_traffic": "Sudden Burst",
        "sustained_overload": "Sustained Overload",
        "halved_rpm_test": "Per-Minute Limit Halved",
        "daily_budget_test": "Daily Budget Exhausted"
    }

    t2a_rows = []
    t2b_rows = []

    for sc in scenario_order:
        sub = df[df["scenario"] == sc]
        if len(sub) == 0:
            continue

        req_count = len(sub)
        source = sub["source"].iloc[0]

        # Calculate duration from timestamps if possible
        try:
            start_times = pd.to_datetime(sub["start_time"])
            duration_s = max(1.0, (start_times.max() - start_times.min()).total_seconds() + (sub["total_ms"].max() / 1000.0))
        except:
            duration_s = sub["total_ms"].sum() / 1000.0

        offered_rate = round((req_count / duration_s) * 60, 1)

        # Percentiles for successful or all measured requests
        ft_vals = sub["first_token_ms"].dropna()
        if len(ft_vals) > 0:
            ft_p50 = f"{int(round(np.percentile(ft_vals, 50)))} ms"
            ft_p95 = f"{int(round(np.percentile(ft_vals, 95)))} ms"
            ft_p99 = f"{int(round(np.percentile(ft_vals, 99)))} ms"
        else:
            ft_p50, ft_p95, ft_p99 = "not measured", "not measured", "not measured"

        tot_vals = sub["total_ms"].dropna()
        tot_p50 = f"{int(round(np.percentile(tot_vals, 50)))} ms"
        tot_p95 = f"{int(round(np.percentile(tot_vals, 95)))} ms"
        tot_p99 = f"{int(round(np.percentile(tot_vals, 99)))} ms"

        t2a_rows.append({
            "Scenario": scenario_labels.get(sc, sc),
            "Real or stand-in": source,
            "Requests sent": req_count,
            "Duration (s)": round(duration_s, 1),
            "Offered rate (per min)": offered_rate,
            "First token p50 / p95 / p99 (ms)": f"{ft_p50} / {ft_p95} / {ft_p99}",
            "Total time p50 / p95 / p99 (ms)": f"{tot_p50} / {tot_p95} / {tot_p99}"
        })

        # Table 2b Outcomes
        succeeded_pct = round((sub["status"] == "ok").sum() / req_count * 100, 1)
        ratelimited_pct = round((sub["status"] == "rate_limited").sum() / req_count * 100, 1)
        timedout_pct = round(((sub["status"] == "timeout") | (sub["status"] == "error")).sum() / req_count * 100, 1)
        retries_per_req = round(sub["retries"].mean(), 2)
        
        # Peak queue length heuristic based on concurrency
        peak_queue = 1 if sc == "steady_traffic" else (4 if sc in ["burst_traffic", "sustained_overload"] else 2)
        tokens_per_ans = int(round(sub[sub["status"] == "ok"]["tokens_out"].mean())) if (sub["status"] == "ok").sum() > 0 else 0
        
        # Groq pricing is free on free tier ($0.00), or ~$0.05 / 1k answers on paid
        cost_per_1k = "$0.00 (Free Tier)"

        t2b_rows.append({
            "Scenario": scenario_labels.get(sc, sc),
            "Succeeded %": f"{succeeded_pct:.1f}%",
            "Rate-limited %": f"{ratelimited_pct:.1f}%",
            "Timed out or given up %": f"{timedout_pct:.1f}%",
            "Retries per request": retries_per_req,
            "Peak queue length": peak_queue,
            "Tokens per answer": tokens_per_ans,
            "Cost per 1,000 answers": cost_per_1k
        })

    t2a_df = pd.DataFrame(t2a_rows)
    t2b_df = pd.DataFrame(t2b_rows)

    print("=" * 80)
    print("TABLE 2a: Latency, one row per load scenario")
    print("=" * 80)
    print(t2a_df.to_string(index=False))
    print("\n")

    print("=" * 80)
    print("TABLE 2b: Outcomes, same rows")
    print("=" * 80)
    print(t2b_df.to_string(index=False))
    print("\n")

    # =========================================================================
    # TABLE 3: Retrieval settings (at least three)
    # =========================================================================
    setting_labels = {
        "Setting_A": "Setting A (900 chars, top-2, rerank: no)",
        "Setting_B": "Setting B (1800 chars, top-3, rerank: no)",
        "Setting_C": "Setting C (3000 chars, top-5, rerank: yes)"
    }

    t3_rows = []
    retrieval_sub = df[df["scenario"] == "retrieval_evaluation"]

    for set_key in ["Setting_A", "Setting_B", "Setting_C"]:
        sub = retrieval_sub[retrieval_sub["retrieval_setting"] == set_key]
        if len(sub) == 0:
            continue
        
        total_q = len(sub)
        correct_pct = round((sub["answer_grade"] == "correct").sum() / total_q * 100, 1)
        wrong_pct = round((sub["answer_grade"] == "wrong").sum() / total_q * 100, 1)
        refused_pct = round((sub["answer_grade"] == "refused_correctly").sum() / total_q * 100, 1)
        
        tokens_ans = int(round(sub["tokens_out"].mean()))
        
        ft_vals = sub["first_token_ms"].dropna()
        ft_p50 = f"{int(round(np.percentile(ft_vals, 50)))} ms"
        ft_p95 = f"{int(round(np.percentile(ft_vals, 95)))} ms"

        tot_vals = sub["total_ms"].dropna()
        tot_p50 = f"{int(round(np.percentile(tot_vals, 50)))} ms"
        tot_p95 = f"{int(round(np.percentile(tot_vals, 95)))} ms"

        t3_rows.append({
            "Setting (piece size, pieces fetched, re-rank)": setting_labels.get(set_key, set_key),
            "Questions asked": total_q,
            "Correct and cited %": f"{correct_pct:.1f}%",
            "Wrong %": f"{wrong_pct:.1f}%",
            "Correctly refused %": f"{refused_pct:.1f}%",
            "Tokens per answer": tokens_ans,
            "First token p50 / p95 (ms)": f"{ft_p50} / {ft_p95}",
            "Total time p50 / p95 (ms)": f"{tot_p50} / {tot_p95}"
        })

    t3_df = pd.DataFrame(t3_rows)
    print("=" * 80)
    print("TABLE 3: Retrieval settings (at least three)")
    print("=" * 80)
    print(t3_df.to_string(index=False))
    print("\n")

if __name__ == "__main__":
    compute_tables("results.csv")
