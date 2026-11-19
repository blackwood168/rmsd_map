import re
from typing import List

import numpy as np
from rdkit import Chem

from rmsd_map.mol_io.fragment import Fragment

ELEMENT_PATTERN = re.compile(r'^([A-Z][a-z]?)')
HEADER_PATTERN = re.compile(r'^(?P<refcode>\S+)\s*\*\*FRAG\*\*\s*\S+(?:\s|$)')
PERIODIC_TABLE = Chem.GetPeriodicTable()
ELEMENTS = frozenset(PERIODIC_TABLE.GetElementSymbol(i) for i in range(1, 119))


def extract_element(atom_name: str) -> str:
    """Extract an element symbol from an atom name such as C21 or O1%."""
    match = ELEMENT_PATTERN.match(atom_name)
    if match:
        return match.group(1)
    return "" if atom_name.isdecimal() else atom_name[:2].strip()


def read_cor_file(filename: str) -> List[Fragment]:
    fragments = []
    frag_id = None
    header_line = None
    elements = []
    coords = []

    def finish_fragment():
        if not coords:
            raise ValueError(f"{filename}:{header_line}: fragment {frag_id} has no atoms")
        fragments.append(Fragment(frag_id, np.asarray(elements, dtype="U2"),
                                  np.asarray(coords, dtype=np.float32)))

    with open(filename) as cor_file:
        for line_number, line in enumerate(cor_file, start=1):
            line = line.strip()
            if not line:
                continue

            if "**FRAG**" in line:
                match = HEADER_PATTERN.match(line)
                if not match:
                    raise ValueError(f"{filename}:{line_number}: invalid **FRAG** header")
                if frag_id is not None:
                    finish_fragment()
                frag_id = match["refcode"]
                header_line = line_number
                elements = []
                coords = []
                continue

            if frag_id is None:
                continue

            location = f"{filename}:{line_number}: fragment {frag_id}"
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"{location}: expected an atom label and three coordinates")
            element = extract_element(fields[0])
            if element not in ELEMENTS:
                raise ValueError(f"{location}: invalid element in atom label {fields[0]!r}")
            try:
                with np.errstate(over="ignore"):
                    xyz = np.asarray(fields[1:4], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"{location}: expected three numeric coordinates") from exc
            if not np.isfinite(xyz).all():
                raise ValueError(f"{location}: coordinates must be finite float32 values")
            elements.append(element)
            coords.append(xyz)

    if frag_id is not None:
        finish_fragment()

    return fragments
