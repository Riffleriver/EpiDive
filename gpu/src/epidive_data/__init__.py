"""Paths to the reference tables bundled with EpiDive."""

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DIR_IDS = str(
    DATA_DIR
    / "10k.NR1550S.full.aln.core0.99.maf002.SNP.Gene.gene.ids.txt"
)
GENE_REF_PATH = str(DATA_DIR / "10k_4352_ref_blastn_results1selected.csv")
GENE_REP65_PATH = str(DATA_DIR / "10k_4352_rep62s_blastn_results1selected.csv")


def reference_paths():
    """Return the three installed reference paths after validating them."""
    paths = {
        "dir_ids": DIR_IDS,
        "gene_ref_path": GENE_REF_PATH,
        "gene_rep65_path": GENE_REP65_PATH,
    }
    missing = [name for name, path in paths.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Bundled EpiDive reference files are missing: " + ", ".join(missing)
        )
    return paths
