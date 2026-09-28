"""Strict minimal PDB reader. mmCIF and hybrid-36 numbering are not supported."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class AtomKey:
    chain: str
    resseq: int
    name: str
    icode: str = ""
    resname: str = ""
    record: str = "ATOM"


def read_pdb(path):
    """Read first model; choose highest-occupancy alternate, blank/A on ties.

    Alternate conformers are selected per atom, not as a jointly packed residue.
    Input for a real task must therefore be pre-cleaned to one conformer.
    """
    candidates, seen = {}, set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in ("ATOM", "HETATM"):
            continue
        try:
            key = AtomKey(line[21:22].strip(), int(line[22:26]), line[12:16].strip(),
                          line[26:27].strip(), line[17:20].strip(), record)
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            occ = float(line[54:60].strip() or "1")
            alt = line[16:17].strip()
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Malformed PDB atom at line {number}") from exc
        if not np.isfinite(xyz).all() or not np.isfinite(occ) or occ < 0:
            raise ValueError(f"Invalid coordinate/occupancy at line {number}")
        if (key, alt) in seen:
            raise ValueError(f"Duplicate atom/conformer at line {number}: {key}")
        seen.add((key, alt))
        # max occupancy, then blank, then A, then lexicographically first.
        rank = (-occ, 0 if alt == "" else 1 if alt == "A" else 2, alt)
        if key not in candidates or rank < candidates[key][0]:
            candidates[key] = (rank, xyz)
    if not candidates:
        raise ValueError("PDB contains no atoms")
    return {key: value[1] for key, value in candidates.items()}


def select_atoms(atoms, selectors):
    result = {}
    for label, selector in selectors.items():
        matches = [xyz for key, xyz in atoms.items()
                   if all(getattr(key, field) == value for field, value in selector.items()
                          if field != "role")]
        if len(matches) != 1:
            raise ValueError(f"Atom {label}: expected one match, found {len(matches)}")
        result[label] = matches[0].copy()
    return result
