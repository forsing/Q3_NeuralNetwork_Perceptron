#!/usr/bin/env python3
"""
Q3 Neural Network — tehnika: Quantum Perceptron (Kapoor-Wiebe-Svore, 2016)
(čisto kvantno, bez klasičnog treniranja i bez hibrida).

Ideja: „učenje težina“ = Grover pretraga nad prostorom binarnih težina w ∈ {0,1}^nq.
Oracle označava težine čija je veza sa binarnom feature-om iz CELOG CSV-a
(⟨w, f⟩ ≥ T) — perceptronsko pravilo okidanja. Grover amplifikuje takve težine;
iz Statevector-a se čita bias i mapira u NEXT rastuću sedmorku ∈ {1..39}.

Sve deterministički: seed=39; feature i prag izvedeni iz CSV-a.

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.quantum_info import Statevector

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

# Deterministička grid-optimizacija (nq, T, Δiter) po meri cos(bias, freq_csv).
GRID_NQ = (6, 7, 8)
GRID_ITER_DELTA = (-1, 0, 1)
# Prag T biramo relativno u odnosu na broj 1-bitova u featuru |f|: T ∈ {|f|-1, |f|, |f|+1}
GRID_T_DELTA = (-1, 0, 1)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    """Histogram pojavljivanja brojeva 1..39 u celom H."""
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def feature_binary(H: np.ndarray, nq: int) -> np.ndarray:
    """
    Deterministički binarni feature f ∈ {0,1}^nq iz CELOG CSV-a:
      - podeli 1..39 u nq blokova jednake širine
      - po bloku uzmi srednju frekvenciju
      - binarizuj: 1 ako je blok iznad medijane blok-srednjih, inače 0
    """
    f = freq_vector(H)
    edges = np.linspace(0, N_MAX, nq + 1, dtype=int)
    blk = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(nq)],
        dtype=np.float64,
    )
    med = float(np.median(blk))
    bits = (blk > med).astype(np.int64)
    # Ako su svi 0 (degenerisan slučaj), podigni najveći na 1.
    if int(bits.sum()) == 0 and blk.size > 0:
        bits[int(np.argmax(blk))] = 1
    return bits


# =========================
# Grover: oracle + diffusion
# =========================
def build_perceptron_oracle(nq: int, f_bits: np.ndarray, T: int) -> QuantumCircuit:
    """
    Perceptronski oracle: -1 na stanjima |w⟩ gde je ⟨w, f⟩ ≥ T, +1 inače.
    Inner product se računa kao popcount(w AND f_mask).
    """
    f_mask = 0
    for i, b in enumerate(f_bits):
        if int(b) == 1:
            f_mask |= (1 << i)
    diag = np.ones(2 ** nq, dtype=complex)
    for w in range(2 ** nq):
        inner = bin(w & f_mask).count("1")
        if inner >= T:
            diag[w] = -1.0 + 0j
    return Diagonal(diag.tolist())


def build_diffusion(nq: int) -> QuantumCircuit:
    qc = QuantumCircuit(nq, name="Diff")
    qc.h(range(nq))
    qc.x(range(nq))
    qc.h(nq - 1)
    if nq >= 2:
        qc.mcx(list(range(nq - 1)), nq - 1)
    else:
        qc.z(0)
    qc.h(nq - 1)
    qc.x(range(nq))
    qc.h(range(nq))
    return qc


def count_marked(nq: int, f_bits: np.ndarray, T: int) -> int:
    f_mask = 0
    for i, b in enumerate(f_bits):
        if int(b) == 1:
            f_mask |= (1 << i)
    m = 0
    for w in range(2 ** nq):
        if bin(w & f_mask).count("1") >= T:
            m += 1
    return m


def optimal_iterations(n: int, m: int) -> int:
    """k* = round((π/4) · √(N/M)), minimum 1."""
    if m <= 0 or n <= 0:
        return 0
    return max(1, int(round((np.pi / 4.0) * np.sqrt(n / m))))


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


def perceptron_probs(nq: int, f_bits: np.ndarray, T: int, k_iter: int) -> np.ndarray:
    oracle = build_perceptron_oracle(nq, f_bits, T)
    diff = build_diffusion(nq)
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for _ in range(max(0, k_iter)):
        qc.compose(oracle, range(nq), inplace=True)
        qc.compose(diff, range(nq), inplace=True)
    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    s = float(p.sum())
    return p / s if s > 0 else p


# =========================
# Determ. grid-optimizacija (nq, T, iter) po meri cos(bias, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    f_csv_n = f_csv / float(f_csv.sum() or 1.0)
    best = None
    for nq in GRID_NQ:
        N_space = 2 ** nq
        f_bits = feature_binary(H, nq)
        fw = int(f_bits.sum())
        for td in GRID_T_DELTA:
            T = max(1, min(fw + td, nq))
            M = count_marked(nq, f_bits, T)
            if M <= 0 or M >= N_space:
                continue
            k_star = optimal_iterations(N_space, M)
            for d in GRID_ITER_DELTA:
                k_iter = max(1, k_star + d)
                try:
                    probs = perceptron_probs(nq, f_bits, T, k_iter)
                    b = bias_39(probs)
                    score = cosine(b, f_csv_n)
                except Exception:
                    continue
                key = (score, -nq, -abs(td), -abs(d))
                if best is None or key > best[0]:
                    best = (
                        key,
                        dict(
                            nq=nq, T=T, k_iter=k_iter, delta=d,
                            fw=fw, M=M, score=score,
                            f_bits=f_bits.copy(),
                        ),
                    )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q3 NN (Quantum Perceptron / KWS): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2

    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| |f|:", best["fw"],
        "| T:", best["T"],
        "| M (označenih):", best["M"],
        "| iter:", best["k_iter"],
        "(Δ vs k*:", best["delta"], ")",
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )
    print("feature f (bitovi):", best["f_bits"].tolist())

    probs = perceptron_probs(best["nq"], best["f_bits"], best["T"], best["k_iter"])
    pred = pick_next_combination(probs)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q3 NN (Quantum Perceptron / KWS): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 8 | |f|: 4 | T: 3 | M (označenih): 80 | iter: 2 (Δ vs k*: 1 ) | cos(bias, freq_csv): 0.994079
feature f (bitovi): [0, 1, 0, 0, 1, 0, 1, 1]
predikcija NEXT: (2, 3, x, y, z, 19, 20)
"""



"""
Q3_NeuralNetwork_Perceptron.py — tehnika: Quantum Perceptron (Kapoor-Wiebe-Svore, 2016)

Učita CEO CSV i iz njega izvede binarni feature f ∈ {0,1}^nq:
Podeli 1..39 u nq blokova, uzme srednju frekvenciju po bloku, binarizuje po medijani (1 iznad medijane, inače 0).
Pretražuje prostor binarnih težina w ∈ {0,1}^nq (veličine 2^nq) preko Grover-a.
Oracle označava težine sa perceptronskim pravilom okidanja: ⟨w, f⟩ ≥ T (popcount nad w AND f_mask).
Grover amplifikacija → Statevector → bias_39 → NEXT.
Deterministička grid-optimizacija (nq, T = |f|+{-1,0,+1}, Δiter oko k*) po meri cos(bias, freq_csv).

Tehnike:
„Učenje težina“ kao Grover pretraga nad w-prostorom (KWS ideja: kvantni perceptron bez klasičnog optimizera).
Fazni Diagonal oracle ručno konstruisan preko inner-product-a (popcount).
Standardni Grover difuzor, formula za k*.
Egzaktni Statevector.

Prednosti:
Literaturno jaka, „pravo kvantno učenje“ (trening kroz Grover, ne gradijent).
Determinističko, brzo, malo parametara.
Perceptronski prag T eksplicitno dozvoljava podešavanje „strogoće“ okidanja.
Feature f je informativniji od puke top-M liste (blok-struktura).

Nedostaci:
Ulazni feature f je svega nq bita — vrlo gruba kompresija celog histograma.
„Težine“ ne nose konkretno značenje za loto — prostor w je apstraktan, pa predikcija zavisi od načina mapiranja w → bias_39.
Oracle je i dalje dijagonalni i uniforman početak — suštinski slično Grover-u iz Q2, samo sa drugačijim markiranjem.
Ne modeluje sekvencu ni kombinacije parova.
Kao i kod Grover-a, mera cos(bias, freq_csv) tera ka reprodukciji frekvencije.
Skalabilnost: 2^nq za Statevector, isto ograničenje.
"""
