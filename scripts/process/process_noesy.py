import argparse
import os
import random
import subprocess
import tempfile
import logging # For better error messages

from Bio.PDB import PDBParser # MMCIFParser might not be needed if input is PDB from NPZ
from Bio.PDB.vectors import Vector # Not directly used by new functions but kept for get_atoms
import numpy as np

logger = logging.getLogger(__name__)

# Standard PDB atom names for backbone and CB (simplified)
# This will be used by map_atom_info_to_pdb_atom_name
STANDARD_ATOM_NAMES = {
    'N': 'N', 'CA': 'CA', 'C': 'C', 'O': 'O', 'CB': 'CB'
}

# Mapping from atomic number to element symbol (common elements in proteins)
ATOMIC_NUMBER_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S',
    # Add more if other elements like P, metals etc. are expected in NPZ 'atoms' array
}


# CASP13 naming conventions for relevant atoms (for NOESY peak picking)
RELEVANT_ATOMS = {
    'ALA': ['CB'],
    'ARG': ['HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'NH1', 'NH2'], # Assuming HB2/3, HG2/3, HD2/3 for methyls if any, HE, NH1, NH2 for amides
    'ASN': ['HB2', 'HB3', 'HD21', 'HD22'], # HD21, HD22 for amide
    'ASP': ['HB2', 'HB3'],
    'CYS': ['HB2', 'HB3', 'HG'], # HG for SH group, not methyl, but often observed
    'GLN': ['HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22'], # HE21, HE22 for amide
    'GLU': ['HB2', 'HB3', 'HG2', 'HG3'],
    'GLY': ['HA2', 'HA3'], # HA2, HA3 often observed
    'HIS': ['HB2', 'HB3', 'HD2', 'HE1'], # Methyls if any, HD2, HE1
    'ILE': ['HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13', 'HG12', 'HG13'], # Methyls: CD1 (HD1*), CG2 (HG2*)
    'LEU': ['HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23'], # Methyls: CD1 (HD1*), CD2 (HD2*)
    'LYS': ['HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'], # HZ* for NH3+
    'MET': ['HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'], # Methyl: CE (HE*)
    'PHE': ['HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'],
    'PRO': ['HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3'],
    'SER': ['HB2', 'HB3', 'HG'],
    'THR': ['HB', 'HG21', 'HG22', 'HG23'], # Methyl: CG2 (HG2*)
    'TRP': ['HB2', 'HB3', 'HD1', 'HE1', 'HZ2', 'HZ3', 'HH2'],
    'TYR': ['HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH'],
    'VAL': ['HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23'], # Methyls: CG1 (HG1*), CG2 (HG2*)
}

# Backbone amide hydrogen (for NOESY peak picking)
BACKBONE_AMIDE = "H"

PDB2PQR_PATH = "/home/swebot/.local/bin/pdb2pqr30" # Path to pdb2pqr executable


# --- New NPZ Processing Functions ---

def parse_npz(npz_file_path: str) -> dict:
    """
    Loads specified arrays from an .npz file.
    Expected keys: 'atoms', 'coords', 'residues', 'chains'.
    """
    try:
        data = np.load(npz_file_path)
        # It's good practice to check if keys exist, but for now, assume they do as per spec.
        npz_data = {
            'atoms': data['atoms'],       # Shape (num_total_atoms, atom_feature_dim) e.g. atom_entry[1] is atomic_num
            'coords': data['coords'],     # Shape (num_total_atoms, 3)
            'residues': data['residues'], # Shape (num_total_residues, residue_feature_dim) e.g., res_entry[0] is resname
            'chains': data['chains']      # Shape (num_chains, chain_feature_dim) e.g., chain_entry[0] is chain_id_str
        }
        # Validate shapes or presence of essential data if necessary
        if not all(key in npz_data for key in ['atoms', 'coords', 'residues', 'chains']):
            raise KeyError("One or more required keys (atoms, coords, residues, chains) missing from NPZ.")
        return npz_data
    except FileNotFoundError:
        logger.error(f"Error: NPZ file not found at {npz_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing NPZ file {npz_file_path}: {e}")
        raise


PDB_ATOM_NAME_FALLBACK_MAX_HEAVY = 5 # Number of initial heavy atoms to try mapping using simple order
COMMON_RESIDUE_HEAVY_ATOM_ORDER = { # Simplified typical order for first few heavy atoms
    0: 'N', 1: 'CA', 2: 'C', 3: 'O', 4: 'CB'
}
GLY_ATOM_ORDER = {0: 'N', 1: 'CA', 2: 'C', 3: 'O'} # Glycine lacks CB

def map_atom_info_to_pdb_atom_name(atom_npz_entry: np.ndarray,
                                   residue_name: str,
                                   atom_idx_in_residue_npz: int,
                                   all_atom_entries_for_this_residue: list) -> str:
    """
    Heuristically maps atom information from NPZ to a PDB atom name.
    This is a simplified heuristic and might need significant refinement for accuracy.

    Args:
        atom_npz_entry: A single row from the NPZ 'atoms' array.
                        Assumes atom_npz_entry[1] is atomic number.
        residue_name (str): 3-letter code of the parent residue.
        atom_idx_in_residue_npz (int): 0-based index of this atom within its residue's
                                       list of atoms as read from NPZ 'atoms' array.
        all_atom_entries_for_this_residue (list): List of all atom_npz_entry for the current residue.
                                                  Used to count heavy atoms.

    Returns:
        str: A 4-character PDB atom name (e.g., " CA ", " CB ", " C1 ").
    """
    atomic_number = atom_npz_entry[1]
    element_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, "X").upper() # Default to X if unknown

    # Count how many heavy atoms appear before this one (or are this one) in the residue's NPZ list
    heavy_atom_counter_for_this_atom = -1
    current_atom_is_heavy = (atomic_number != 1) # Hydrogen is 1

    for i, atm_entry in enumerate(all_atom_entries_for_this_residue):
        if atm_entry[1] != 1: # if it's a heavy atom
            heavy_atom_counter_for_this_atom +=1
        if i == atom_idx_in_residue_npz: # Found the current atom
            break

    if not current_atom_is_heavy: # If it's a Hydrogen
        # Basic hydrogen naming: H plus a number, or HN, HA etc. if logic was more complex
        # PDB2PQR will rename these anyway. For now, generic.
        # To make it somewhat unique for pdb2pqr, use its index within the residue.
        return f"{element_symbol:<2}{str(atom_idx_in_residue_npz + 1):<2}"[:4].ljust(4)


    # For heavy atoms, try predefined names for backbone/CB
    atom_order_map = GLY_ATOM_ORDER if residue_name == "GLY" else COMMON_RESIDUE_HEAVY_ATOM_ORDER

    if heavy_atom_counter_for_this_atom < PDB_ATOM_NAME_FALLBACK_MAX_HEAVY:
        pdb_name_stem = atom_order_map.get(heavy_atom_counter_for_this_atom)
        if pdb_name_stem:
            # PDB atom names are 4 chars. Element right-justified if name is short.
            # e.g., " CA ", " O  ", " CB "
            if len(pdb_name_stem) == 1 and element_symbol != pdb_name_stem : # Like 'O' vs ' OXT'
                 return f" {pdb_name_stem}  "[:4] # Pad to 4, e.g. " O  "
            elif len(pdb_name_stem) == 1 and element_symbol == pdb_name_stem:
                 return f"{element_symbol:<2}  "[:4].ljust(4) # "C   " for CA if element is C
            elif len(pdb_name_stem) == 2: # e.g. CA, CB
                 return f" {pdb_name_stem:<2}"[:4].ljust(4) # " CA ", " CB "

    # Fallback for other heavy atoms or if order doesn't match
    # Use element symbol and a number based on its appearance order among heavy atoms in the residue
    # e.g. C1, C2, N1 etc. This ensures uniqueness within the residue for PDB.
    # PDB format: Element right justified if name is short (e.g. " C1 ", "S G ")
    # Atom name: columns 13-16
    # Element : columns 77-78
    # For atom name, it's more like "CG1 ", "SD  "
    # Let's try element + count relative to its element type in this residue

    # Count occurrences of this element type up to this atom in the residue
    element_type_count = 0
    for i, atm_entry in enumerate(all_atom_entries_for_this_residue):
        if atm_entry[1] == atomic_number: # Same element
            element_type_count += 1
        if i == atom_idx_in_residue_npz:
            break

    # Name like "C1", "C2", "N1" etc. Pad appropriately.
    # PDB format requires atom names to be unique within a residue.
    # Element symbol is usually left-justified in the 2-char space for element in ATOM record.
    # Atom name field (4 chars): " CA ", " HG1" (if H is gamma on C1)
    # Element (2 chars, cols 77-78): " C", " H"
    # Heuristic: EL+num, e.g. C1, N1. For PDB, this might be "C1  " or " C1 ".
    # Let's try: " CG1", " OE2" type format if possible, else EL+num.
    # This simple heuristic can't do that well.
    # Fallback: ElementSymbol + sequential number of that element type in residue
    name_stem = f"{element_symbol}{element_type_count}" # e.g. C1, C2, N1
    if len(name_stem) == 2:
        return f" {name_stem:<2}"[:4].ljust(4) # " C1 ", " N1 "
    elif len(name_stem) == 3:
        return f" {name_stem}"[:4].ljust(4)   # " C10" (if many carbons)
    elif len(name_stem) == 1: # Should not happen for heavy + number
        return f" {name_stem}  "[:4].ljust(4)
    else: # too long
        return name_stem[:4].ljust(4)


def write_temp_pdb_from_npz(npz_data: dict, temp_pdb_path: str):
    """
    Writes a temporary PDB file from parsed NPZ data.
    Uses heuristic atom naming.

    NPZ data structure assumptions (based on user description and typical formats):
    - npz_data['atoms']: (total_atoms, features_per_atom); atom_entry[1] is atomic_number.
    - npz_data['coords']: (total_atoms, 3)
    - npz_data['residues']: (total_residues_in_protein, features_per_residue)
        - res_entry[0]: 3-letter residue name (str)
        - res_entry[2]: chain index (int, 0-based) in npz_data['chains']
        - res_entry[3]: residue sequence number within its chain (int, 0-based for processing, converted to 1-based for PDB)
        - res_entry[4]: global start index for this residue's atoms in 'atoms'/'coords' arrays
        - res_entry[5]: number of atoms this residue has in 'atoms'/'coords' arrays
    - npz_data['chains']: (num_chains, features_per_chain)
        - chain_entry[0]: chain ID (str, e.g., 'A')
    """
    atom_serial = 0
    with open(temp_pdb_path, 'w') as pdb_file:
        for res_idx_in_protein, res_entry in enumerate(npz_data['residues']):
            res_name = str(res_entry[0]) # Ensure it's string, e.g. 'VAL'
            chain_npz_idx = int(res_entry[2])
            # PDB residue numbers are 1-indexed. Assuming res_entry[3] is 0-indexed seq num in chain.
            res_seq_num_pdb = int(res_entry[3]) + 1

            atom_start_global_idx = int(res_entry[4])
            num_atoms_in_this_res_npz = int(res_entry[5])

            chain_id_str = str(npz_data['chains'][chain_npz_idx][0]) if chain_npz_idx < len(npz_data['chains']) else 'A'
            if not chain_id_str.strip() or len(chain_id_str) > 1: # Default or handle multi-char chain IDs if necessary
                chain_id_pdb = 'A'
            else:
                chain_id_pdb = chain_id_str.strip()

            # Get all atom entries for the current residue to help with naming
            current_residue_atom_npz_entries = [
                npz_data['atoms'][atom_start_global_idx + i] for i in range(num_atoms_in_this_res_npz)
            ]

            for atom_idx_in_res, atom_npz_entry_original in enumerate(current_residue_atom_npz_entries):
                atom_serial += 1
                global_atom_idx = atom_start_global_idx + atom_idx_in_res

                coords = npz_data['coords'][global_atom_idx]
                x, y, z = coords[0], coords[1], coords[2]

                # Use the heuristic naming function
                # Pass the original atom_npz_entry (from global 'atoms' array)
                atom_pdb_name = map_atom_info_to_pdb_atom_name(
                    atom_npz_entry_original,
                    res_name,
                    atom_idx_in_res, # 0-indexed position of this atom within this residue's list from NPZ
                    current_residue_atom_npz_entries
                )

                atomic_number = atom_npz_entry_original[1]
                element_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, "X").upper()
                # PDB format: element symbol is right-justified in columns 77-78
                element_symbol_pdb = f"{element_symbol:>2}"

                # ATOM record format string (fixed width)
                # Record name, Atom serial, Atom name, Alt loc, Res name, Chain ID, Res seq num, Insertion, X, Y, Z, Occupancy, Temp factor, Element, Charge
                # "ATOM  %5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  "
                # Atom name: %-4s (left justified) or %4s. PDB standard is complex.
                # For atom name like " CA ", use atom_pdb_name directly if it's already 4 chars.
                # If atom_pdb_name is "CA", it should be " CA ".
                # The map_atom_info_to_pdb_atom_name should return a 4-char string.

                pdb_line = (
                    f"ATOM  {atom_serial:5d} {atom_pdb_name:4s} {res_name:3s} {chain_id_pdb:1s}{res_seq_num_pdb:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          "
                    f"{element_symbol_pdb:2s}  \n"
                )
                pdb_file.write(pdb_line)
    if atom_serial == 0:
        logger.warning(f"No atoms were written to temporary PDB file: {temp_pdb_path}. NPZ parsing might have issues.")

# --- End of New NPZ Processing Functions ---


def add_hydrogens(pdb_file, output_pdb_file):
    """
    Adds hydrogen atoms to a PDB file using PDB2PQR.
    Shells out to the pdb2pqr30 command-line tool.
    """
    try:
        # pdb2pqr30 --ff=AMBER --titration-state-method=propka --with-ph=7.0 <input_pdb> <output_pqr>
        # We want PDB output, so we use a temporary pqr file and then convert, or see if pdb2pqr can output PDB directly
        # Looking at pdb2pqr help, it seems it outputs PQR format.
        # For this task, we'll assume that PDB2PQR adds hydrogens and the output can be parsed by BioPython.
        # The --pdb-output flag can be used with pdb2pqr30 for compatible versions.
        # If not, a conversion step might be needed or careful parsing of PQR.
        # Let's try with --pdb-output first.
        command = [
            PDB2PQR_PATH,
            "--ff=AMBER",  # Force field
            "--pdb-output", # Request PDB output with hydrogens
            pdb_file,
            output_pdb_file
        ]
        print(f"Running PDB2PQR: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"PDB2PQR stdout: {result.stdout}")
        if result.stderr:
            print(f"PDB2PQR stderr: {result.stderr}")
        if not os.path.exists(output_pdb_file):
            raise FileNotFoundError(f"PDB2PQR did not generate the output file: {output_pdb_file}")
        print(f"Successfully added hydrogens: {output_pdb_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running PDB2PQR for {pdb_file}: {e}")
        print(f"PDB2PQR stdout: {e.stdout}")
        print(f"PDB2PQR stderr: {e.stderr}")
        raise  # Re-raise the exception to be handled by the caller
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # This might happen if PDB2PQR_PATH is incorrect or pdb2pqr30 is not executable
        print(f"Please ensure {PDB2PQR_PATH} is correct and executable.")
        raise

def get_atoms(structure):
    """
    Extracts relevant atoms (backbone amide HN, sidechain methyl groups, Asn/Gln amide groups)
    from a BioPython Structure object based on CASP13 naming conventions.
    """
    relevant_atoms_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                # Skip non-standard residues or those not in our list
                if res_name not in RELEVANT_ATOMS and res_name != 'GLY': # GLY handled by backbone H
                    # Could add a check for common modified residues if needed
                    continue

                for atom in residue:
                    atom_name = atom.get_name()
                    # Check for backbone amide hydrogen
                    if atom_name == BACKBONE_AMIDE and atom.element == 'H' and residue.has_id('N') and atom.get_parent().id == 'N':
                         # Ensure it's bonded to N for amides (common case for PDB2PQR output)
                        relevant_atoms_list.append(atom)
                        continue # Move to next atom once identified as backbone H

                    # Check for sidechain atoms based on RELEVANT_ATOMS map
                    if res_name in RELEVANT_ATOMS:
                        if atom_name in RELEVANT_ATOMS[res_name]:
                            relevant_atoms_list.append(atom)
    return relevant_atoms_list

def generate_noesy_data(pdb_file_with_hydrogens, distance_cutoff=5.0):
    """
    Generates NOESY data from a PDB file with hydrogens.
    - Parses the PDB file.
    - Extracts relevant atoms.
    - Calculates distances between atom pairs.
    - Filters pairs by distance_cutoff.
    - Generates NOESY data string.
    - Introduces noise.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_file_with_hydrogens)
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file_with_hydrogens}: {e}")
        return [] # Return empty list on error

    relevant_atoms = get_atoms(structure)
    if not relevant_atoms:
        print(f"No relevant atoms found in {pdb_file_with_hydrogens}")
        return []

    noesy_lines = []
    peak_id_counter = 1

    # Get all residue identifiers once to help with noise generation
    all_residue_ids = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_id()[0] == ' ': # Standard residue
                    all_residue_ids.append(f"{res.get_parent().id}{res.id[1]}") # ChainIDResNum format like A10

    # Calculate distances and generate initial NOESY data
    for i in range(len(relevant_atoms)):
        for j in range(i + 1, len(relevant_atoms)): # Avoid self-pairs and duplicate pairs
            atom1 = relevant_atoms[i]
            atom2 = relevant_atoms[j]

            # Ensure atoms are from different residues
            res1 = atom1.get_parent()
            res2 = atom2.get_parent()
            if res1 == res2:
                continue

            distance = atom1 - atom2 # Bio.PDB.Atom.__sub__ gives distance

            if distance <= distance_cutoff:
                res1_id = f"{res1.get_parent().id}{res1.id[1]}" # ChainIDResNum e.g. A10
                res2_id = f"{res2.get_parent().id}{res2.id[1]}" # ChainIDResNum e.g. B20
                atom1_name = atom1.get_name()
                atom2_name = atom2.get_name()

                noesy_lines.append(f"{res1_id} {res2_id} {peak_id_counter} {distance:.2f} {atom1_name} {atom2_name}")
                peak_id_counter += 1

    # Introduce noise: For ~10% of `residueFrom` entries, add 1-2 incorrect `residueTo` options
    # This is a simple noise model. More sophisticated models could be developed.
    noisy_lines = list(noesy_lines) # Start with a copy of correct lines
    if not all_residue_ids or not noesy_lines: # No data to add noise to
        return noesy_lines

    num_entries_to_add_noise_to = int(0.1 * len(noesy_lines))

    if num_entries_to_add_noise_to == 0 and len(noesy_lines) > 0: # ensure at least one noisy entry if possible
        num_entries_to_add_noise_to = 1

    # Get unique residueFrom atomFrom pairs to select for noise introduction
    unique_from_peaks_indices = []
    seen_from_pairs = set()
    for idx, line in enumerate(noesy_lines):
        parts = line.split()
        res_from = parts[0]
        atom_from = parts[4]
        if (res_from, atom_from) not in seen_from_pairs:
            unique_from_peaks_indices.append(idx)
            seen_from_pairs.add((res_from, atom_from))

    if not unique_from_peaks_indices:
        return noesy_lines # Should not happen if noesy_lines is not empty

    indices_to_corrupt = random.sample(unique_from_peaks_indices, min(num_entries_to_add_noise_to, len(unique_from_peaks_indices)))

    for original_line_idx in indices_to_corrupt:
        original_line_parts = noesy_lines[original_line_idx].split()
        res_from = original_line_parts[0]
        # correct_res_to = original_line_parts[1] # Not used directly for picking noise target
        original_peak_id = original_line_parts[2] # Use same peak ID for plausible ambiguity
        original_distance = float(original_line_parts[3])
        atom_from = original_line_parts[4]
        # original_atom_to = original_line_parts[5] # Not used for noise target atom

        num_false_options = random.randint(1, 2)
        for _ in range(num_false_options):
            # Pick a random incorrect residue (not res_from and not the original res_to for this specific peak)
            incorrect_res_to_candidate = random.choice(all_residue_ids)
            while incorrect_res_to_candidate == res_from : # or incorrect_res_to_candidate == correct_res_to (this check is too restrictive for general noise)
                incorrect_res_to_candidate = random.choice(all_residue_ids)

            # Create a plausible but incorrect atomTo (e.g. a common proton name)
            # For simplicity, let's pick a random atom name from the RELEVANT_ATOMS list, or just 'H'
            random_res_type_for_atom_name = random.choice(list(RELEVANT_ATOMS.keys()))
            if RELEVANT_ATOMS[random_res_type_for_atom_name]:
                incorrect_atom_to = random.choice(RELEVANT_ATOMS[random_res_type_for_atom_name])
            else: # Fallback if a residue type has no specific atoms listed (e.g. GLY only has 'H' from backbone)
                incorrect_atom_to = "H"

            # Generate a slightly perturbed distance
            noisy_distance = max(1.8, original_distance + random.uniform(-1.0, 1.0)) # Ensure distance is somewhat realistic

            noisy_lines.append(f"{res_from} {incorrect_res_to_candidate} {original_peak_id} {noisy_distance:.2f} {atom_from} {incorrect_atom_to}")
            # Note: This might create duplicate peak IDs for the *same* res_from, atom_from but different res_to. This is intended to model ambiguity.

    return noisy_lines

def main():
    """
    Main function to orchestrate the NOESY data generation process from NPZ files.
    """
    parser = argparse.ArgumentParser(description="Generate NOESY-like data from protein structure NPZ files.")
    parser.add_argument("input_dir", help="Directory containing input NPZ files.")
    parser.add_argument("output_dir", help="Directory to save generated NOESY data.")
    parser.add_argument("--distance_cutoff", type=float, default=5.0, help="Distance cutoff for NOESY contacts (Angstroms).")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")

    npz_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".npz")]

    if not npz_files:
        logger.warning(f"No .npz files found in {args.input_dir}")
        return

    for npz_filename in npz_files:
        npz_file_path = os.path.join(args.input_dir, npz_filename)
        base_name_npz = os.path.basename(npz_file_path)
        name_part_npz, _ = os.path.splitext(base_name_npz)

        logger.info(f"\nProcessing {npz_file_path}...")

        temp_initial_pdb_file = None
        temp_hydro_pdb_file = None

        try:
            # 1. Parse NPZ
            npz_data = parse_npz(npz_file_path)

            # 2. Create a temporary PDB file from NPZ data
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp_pdb_initial:
                temp_initial_pdb_file = tmp_pdb_initial.name

            write_temp_pdb_from_npz(npz_data, temp_initial_pdb_file)
            logger.info(f"Generated initial PDB: {temp_initial_pdb_file}")

            if not os.path.exists(temp_initial_pdb_file) or os.path.getsize(temp_initial_pdb_file) == 0:
                logger.error(f"Initial PDB generation failed or produced an empty file for {npz_filename}, skipping.")
                continue

            # 3. Add hydrogens using PDB2PQR to the initial PDB
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb", delete=False) as tmp_pdb_h:
                 temp_hydro_pdb_file = tmp_pdb_h.name

            add_hydrogens(temp_initial_pdb_file, temp_hydro_pdb_file)
            logger.info(f"Generated PDB with hydrogens: {temp_hydro_pdb_file}")

            if not os.path.exists(temp_hydro_pdb_file) or os.path.getsize(temp_hydro_pdb_file) == 0:
                logger.error(f"Hydrogen addition failed or produced an empty file for {npz_filename}, skipping.")
                continue

            # 4. Generate NOESY data from the hydrogenated PDB
            noesy_data = generate_noesy_data(temp_hydro_pdb_file, args.distance_cutoff)

            if noesy_data:
                output_noesy_filename = os.path.join(args.output_dir, f"{name_part_npz}_noesy.txt")
                with open(output_noesy_filename, "w") as f_out:
                    for line in noesy_data:
                        f_out.write(line + "\n")
                logger.info(f"Generated NOESY data for {base_name_npz} at {output_noesy_filename}")
            else:
                logger.info(f"No NOESY data generated for {base_name_npz}.")

        except Exception as e:
            logger.error(f"Error processing file {npz_file_path}: {e}", exc_info=True)
        finally:
            # Clean up temporary files
            if temp_initial_pdb_file and os.path.exists(temp_initial_pdb_file):
                os.remove(temp_initial_pdb_file)
                logger.debug(f"Removed temp initial PDB: {temp_initial_pdb_file}")
            if temp_hydro_pdb_file and os.path.exists(temp_hydro_pdb_file):
                os.remove(temp_hydro_pdb_file)
                logger.debug(f"Removed temp hydrogenated PDB: {temp_hydro_pdb_file}")

    logger.info("\nProcessing complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
