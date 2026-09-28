from dataclasses import dataclass
import math
from .measure import distance, angle, dihedral, angular_difference

FUNCTIONS = {"distance": distance, "angle": angle, "dihedral": dihedral}
ARITY = {"distance": 2, "angle": 3, "dihedral": 4}


@dataclass(frozen=True)
class Constraint:
    name: str
    kind: str
    atoms: tuple
    target: float
    tolerance: float
    scale: float
    weight: float = 1.0

    def __post_init__(self):
        if self.kind not in FUNCTIONS or len(self.atoms) != ARITY[self.kind]:
            raise ValueError("Invalid geometry kind or atom count")
        if len(set(self.atoms)) != len(self.atoms):
            raise ValueError("Constraint atoms must be distinct labels")
        if not all(math.isfinite(x) for x in (self.target, self.tolerance, self.scale, self.weight)):
            raise ValueError("Constraint parameters must be finite")
        if self.tolerance < 0 or self.scale <= 0 or self.weight <= 0:
            raise ValueError("Need tolerance >= 0, scale > 0, weight > 0")
        if self.kind == "distance" and self.target < 0:
            raise ValueError("Distance target cannot be negative")
        if self.kind == "angle" and not 0 <= self.target <= 180:
            raise ValueError("Angle target must be in [0, 180]")
        if self.kind == "dihedral" and not -180 <= self.target <= 180:
            raise ValueError("Dihedral target must be in [-180, 180]")

    def measure(self, coordinates):
        try:
            measured = FUNCTIONS[self.kind](*(coordinates[name] for name in self.atoms))
        except KeyError as exc:
            raise ValueError(f"Missing required atom: {exc.args[0]}") from exc
        delta = abs(angular_difference(measured, self.target)) if self.kind == "dihedral" else abs(measured - self.target)
        violation = max(0.0, delta - self.tolerance) / self.scale
        return measured, violation


@dataclass(frozen=True)
class Theozyme:
    name: str
    atoms: dict
    constraints: tuple
    description: str = ""

    @classmethod
    def from_dict(cls, data):
        if set(data) - {"name", "description", "atoms", "constraints"}:
            raise ValueError("Unknown task fields")
        atoms = data["atoms"]
        if not atoms:
            raise ValueError("Task requires atoms")
        allowed = {"chain", "resseq", "name", "icode", "resname", "record", "role"}
        for label, atom in atoms.items():
            if set(atom) - allowed or not {"chain", "resseq", "name", "role"} <= set(atom):
                raise ValueError(f"Invalid selector for {label}")
            if atom["role"] not in ("fixed", "generated"):
                raise ValueError("Every atom must be fixed or generated")
            if type(atom["resseq"]) is not int:
                raise ValueError("resseq must be an integer")
            if atom.get("record", "ATOM") not in ("ATOM", "HETATM"):
                raise ValueError("record must be ATOM or HETATM")
        constraints = tuple(Constraint(**{**c, "atoms": tuple(c["atoms"])}) for c in data["constraints"])
        if not constraints or len({c.name for c in constraints}) != len(constraints):
            raise ValueError("Need nonempty, uniquely named constraints")
        if any(set(c.atoms) - set(atoms) for c in constraints):
            raise ValueError("Constraint references an undefined atom")
        return cls(data["name"], atoms, constraints, data.get("description", ""))

    def is_steerable(self, constraint):
        return any(self.atoms[a]["role"] == "generated" for a in constraint.atoms)

    def validate_for_steering(self):
        if not any(self.is_steerable(c) for c in self.constraints):
            raise ValueError("All scored constraints involve fixed atoms only: no steering signal")
