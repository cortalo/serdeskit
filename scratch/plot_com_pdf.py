"""Plot matlab_golden/data/com_pdf_c2c_thru.csv -- MATLAB COM3.70's own
combined interference+noise PDF (Equation 93A-45), same "PDF" semilogy
plot com_ieee8023_93a_370.m:445 produces.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parents[1]
DATA_PATH = REPO_ROOT / "matlab_golden" / "data" / "com_pdf_c2c_thru.csv"


def load(path: Path) -> tuple[list[float], list[float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = [(float(row["y"]), float(row["pdf"])) for row in reader]
    y, pdf = zip(*rows)
    return list(y), list(pdf)


def main() -> None:
    y, pdf = load(DATA_PATH)

    _, ax = plt.subplots()
    ax.semilogy(y, pdf)
    ax.set_xlabel("y (V)")
    ax.set_ylabel("probability")
    ax.set_title("PDF (C2C thru, combined interference + noise)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
