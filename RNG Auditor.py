import secrets
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

# 1. Game Math Configuration (3-Reel Slot Machine)
SYMBOLS = ["Cherry", "Lemon", "Bell", "Diamond", "Seven"]
PROBS = [0.40, 0.30, 0.15, 0.10, 0.05]  # Must sum to 1.0
PAYOUTS = {"Cherry": 2, "Lemon": 5, "Bell": 15, "Diamond": 40, "Seven": 100}


def get_theoretical_rtp() -> float:
    """Calculate exact theoretical Return to Player (Expected Value)."""
    return sum((p ** 3) * PAYOUTS[sym] for sym, p in zip(SYMBOLS, PROBS))


def spin_compliant(n_spins: int) -> np.ndarray:
    """Simulate spins using cryptographically seeded PRNG."""
    rng = np.random.default_rng(secrets.randbits(128))
    return rng.choice(SYMBOLS, size=(n_spins, 3), p=PROBS)


def spin_biased_weights(n_spins: int) -> np.ndarray:
    """Defect 1: Subtle symbol weight skew (suppresses 'Seven' by 1%)."""
    bugged_probs = [0.405, 0.305, 0.15, 0.10, 0.04]
    rng = np.random.default_rng(42)
    return rng.choice(SYMBOLS, size=(n_spins, 3), p=bugged_probs)


def spin_sticky_rng(n_spins: int) -> np.ndarray:
    """Defect 2: Serial correlation bug (15% chance to repeat previous spin)."""
    base_spins = spin_compliant(n_spins)
    rng = np.random.default_rng(99)
    for i in range(1, n_spins):
        if rng.random() < 0.15:
            base_spins[i] = base_spins[i - 1]
    return base_spins


def run_chi_square_test(spins: np.ndarray, alpha: float = 0.01):
    """Test 1: Goodness-of-Fit for symbol frequency uniformity."""
    flat_spins = spins.flatten()
    total_symbols = len(flat_spins)
    observed = [int(np.sum(flat_spins == sym)) for sym in SYMBOLS]
    expected = [p * total_symbols for p in PROBS]

    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    status = "PASS" if p_value > alpha else "FAIL"
    return round(float(chi2_stat), 4), round(float(p_value), 6), status, observed, expected


def run_wald_wolfowitz_test(spins: np.ndarray, alpha: float = 0.01):
    """Test 2: Runs Test for sequential independence on Win/Loss binary sequence."""
    wins = np.zeros(len(spins), dtype=int)
    for sym in PAYOUTS:
        wins[np.all(spins == sym, axis=1)] = 1

    # Cast to float to prevent 32-bit integer overflow on large sample sizes
    n1 = float(np.sum(wins == 1))
    n0 = float(np.sum(wins == 0))
    n = n1 + n0

    runs = float(1 + np.sum(wins[1:] != wins[:-1]))

    expected_runs = ((2.0 * n1 * n0) / n) + 1.0
    var_runs = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / ((n ** 2) * (n - 1.0))
    z_stat = (runs - expected_runs) / np.sqrt(var_runs)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    status = "PASS" if p_value > alpha else "FAIL"
    return round(float(z_stat), 4), round(float(p_value), 6), status


def evaluate_rtp(spins: np.ndarray, theo_rtp: float, confidence: float = 0.99):
    """Test 3: Empirical RTP vs Theoretical RTP using Central Limit Theorem."""
    wins = np.zeros(len(spins), dtype=float)
    for sym, mult in PAYOUTS.items():
        wins[np.all(spins == sym, axis=1)] = mult

    empirical_rtp = float(np.mean(wins))
    std_err = float(stats.sem(wins))
    ci_low, ci_high = stats.norm.interval(confidence, loc=empirical_rtp, scale=std_err)

    status = "PASS" if ci_low <= theo_rtp <= ci_high else "FAIL"
    return (
        round(empirical_rtp * 100, 4),
        round(float(ci_low) * 100, 4),
        round(float(ci_high) * 100, 4),
        status,
    )


def build_math_model_df() -> pd.DataFrame:
    """Build reference table of theoretical probabilities and RTP contributions."""
    rows = []
    for sym, p in zip(SYMBOLS, PROBS):
        hit_prob = p ** 3
        payout = PAYOUTS[sym]
        rtp_contrib = hit_prob * payout
        rows.append({
            "Symbol": sym,
            "Single Reel Probability": p,
            "3-of-a-Kind Hit Frequency": round(hit_prob, 6),
            "1-in-N Hit Rate": round(1 / hit_prob, 1),
            "Payout Multiplier": f"{payout}x",
            "RTP Contribution (%)": round(rtp_contrib * 100, 4),
        })
    return pd.DataFrame(rows)


def main():
    n_spins = 500_000
    theo_rtp = get_theoretical_rtp()
    builds = [
        ("Build v1.0 (Baseline RNG)", spin_compliant),
        ("Build v1.1 (Biased Symbol Weights)", spin_biased_weights),
        ("Build v1.2 (Sticky/Correlated RNG)", spin_sticky_rng),
    ]

    audit_rows = []
    freq_rows = []

    print("=== SLOT RNG & PAYOUT COMPLIANCE SIMULATION ===")
    print(f"Sample Size: {n_spins:,} spins | Theoretical RTP: {theo_rtp:.4%}\n")

    for build_name, spin_func in builds:
        data = spin_func(n_spins)
        chi2, chi_p, chi_status, observed, expected = run_chi_square_test(data)
        z_stat, runs_p, runs_status = run_wald_wolfowitz_test(data)
        emp_rtp, ci_low, ci_high, rtp_status = evaluate_rtp(data, theo_rtp)

        overall = "APPROVED" if all(s == "PASS" for s in [chi_status, runs_status, rtp_status]) else "REJECTED"

        print(f"Target: {build_name} -> [{overall}]")
        print(f"  1. Chi-Square Distribution : stat={chi2}, p={chi_p} [{chi_status}]")
        print(f"  2. Wald-Wolfowitz Runs Test: z={z_stat}, p={runs_p} [{runs_status}]")
        print(f"  3. Empirical RTP (99% CI)  : {emp_rtp}% ({ci_low}% - {ci_high}%) [{rtp_status}]\n")

        audit_rows.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Build Version": build_name,
            "Sample Size": n_spins,
            "Theoretical RTP (%)": round(theo_rtp * 100, 4),
            "Empirical RTP (%)": emp_rtp,
            "99% CI Bounds (%)": f"[{ci_low}%, {ci_high}%]",
            "RTP Status": rtp_status,
            "Chi-Square Stat": chi2,
            "Chi-Square p-value": chi_p,
            "Distribution Status": chi_status,
            "Runs Test Z-Score": z_stat,
            "Runs Test p-value": runs_p,
            "Independence Status": runs_status,
            "Final Verdict": overall,
        })

        for sym, obs, exp in zip(SYMBOLS, observed, expected):
            diff_pct = ((obs - exp) / exp) * 100
            freq_rows.append({
                "Build Version": build_name,
                "Symbol": sym,
                "Expected Count": int(exp),
                "Observed Count": obs,
                "Variance (%)": round(diff_pct, 4),
            })

    # Export multi-sheet Excel report
    output_file = "Compliance_Audit_Report.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        pd.DataFrame(audit_rows).to_excel(writer, index=False, sheet_name="Audit_Results")
        pd.DataFrame(freq_rows).to_excel(writer, index=False, sheet_name="Symbol_Frequencies")
        build_math_model_df().to_excel(writer, index=False, sheet_name="Game_Math_Model")

    print(f"Audit report exported successfully to: {output_file}")


if __name__ == "__main__":
    main()
