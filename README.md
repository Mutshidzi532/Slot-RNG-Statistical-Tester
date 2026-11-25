# Slot-RNG-Statistical-Tester
Python simulation and statistical testing suite for verifying slot machine RNG uniformity, serial independence, and theoretical RTP using Chi-Square, Runs, and CI tests.
# Slot RNG & Payout Compliance Simulator

A Python script for testing Random Number Generator (RNG) behavior and Return to Player (RTP) math on a simulated 3-reel slot machine. 

I built this project to apply statistical hypothesis testing—specifically Goodness-of-Fit, Runs tests, and Confidence Intervals—to a practical software verification problem: checking whether a game engine behaves according to its mathematical spec and catching subtle bugs when it doesn't.

## Why this project?
When testing probabilistic software, a simple average isn't enough. A bugged RNG can produce the right overall payout percentage while still having skewed symbol weights or predictable streaks. This script runs 500,000 simulated spins across three different engine builds to see which statistical tests catch which bugs:

1. **Build v1.0 (Baseline):** Uses Python's `secrets` module to seed NumPy's generator. Should pass all checks.
2. **Build v1.1 (Weight Skew):** Reduces the probability of the highest-paying symbol (`Seven`) by 1% and redistributes it to lower symbols. 
3. **Build v1.2 (Sticky State):** Introduces a 15% chance that a spin repeats the previous outcome. The overall symbol frequencies stay normal, so a basic frequency check misses it, while a serial independence test flags it.

## Statistical Tests Used

* **Empirical RTP vs. Theoretical RTP (99% CI):** Calculates the exact expected value from the paytable and checks whether the simulated sample mean falls inside a 99% normal confidence interval (α = 0.01).
* **Chi-Square Goodness-of-Fit:** Compares observed reel symbol counts against expected theoretical frequencies to detect weighted or biased outcomes.
* **Wald–Wolfowitz Runs Test:** Converts the spin history into a binary Win/Loss sequence and counts consecutive runs to check for serial correlation (clumping or alternating patterns).

## Setup & Usage

Requires Python 3.10+ and the following packages:

```bash
pip install numpy scipy pandas openpyxl
```

Run the simulation:

```bash
python rng_auditor.py
```

### Output
1. **Console Summary:** Prints the test statistics, p-values, and `PASS`/`FAIL` status for each build.
2. **Excel Report (`Compliance_Audit_Report.xlsx`):** Exports the results into a spreadsheet with sheets for the summary metrics, symbol frequency breakdowns, and the underlying game math table.

## Project Structure

```text
├── rng_auditor.py                 # Main simulation and statistical test suite
├── Compliance_Audit_Report.xlsx   # Sample output report from a 500k spin run
└── README.md
```

## Notes / Limitations
* **Overflow handling:** At high spin counts ($N \ge 500,000$), intermediate integer multiplications in the Wald–Wolfowitz variance formula exceed 32-bit limits, so counts are explicitly cast to 64-bit floats before calculation.
* **Sample size sensitivity:** With 500,000 spins, very small biases are easily caught at $\alpha = 0.01$, though smaller sample sizes (e.g., 10,000 spins) have noticeably wider RTP confidence intervals due to slot volatility.
