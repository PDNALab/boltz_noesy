import argparse
import os
import random
import subprocess
import tempfile
import logging # For better error messages
import traceback # For detailed error logging
from pathlib import Path # Ensure Path is imported
import shutil # For copying debug PDB files
# Re-submission to ensure code sync for chain_processed_atom_count initialization.

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


def add_hydrogens(pdb_file: str, output_pdb_file: str) -> bool:
    """
    Adds hydrogen atoms to a PDB file using PDB2PQR.
    Shells out to the pdb2pqr30 command-line tool.
    Returns True on success, False on failure.
    """
    print(f"DEBUG: Entering add_hydrogens function for input: {pdb_file}, output: {output_pdb_file}", flush=True)
    command = [
        PDB2PQR_PATH,
        "--ff=AMBER",      # Force field
        "--pdb-output",    # Request PDB output with hydrogens
        pdb_file,
        output_pdb_file
    ]

    logger.info(f"Calling pdb2pqr30 for {pdb_file}...")
    logger.info(f"Command: {' '.join(command)}")

    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout
            check=False   # Do not raise exception on non-zero exit
        )

        if completed_process.returncode != 0:
            logger.error(f"pdb2pqr30 failed for {pdb_file} with return code {completed_process.returncode}")
            logger.error(f"pdb2pqr30 stdout:\n{completed_process.stdout}")
            logger.error(f"pdb2pqr30 stderr:\n{completed_process.stderr}")
            return False
        else:
            logger.info(f"pdb2pqr30 completed successfully for {pdb_file}.")
            if not os.path.exists(output_pdb_file) or os.path.getsize(output_pdb_file) == 0:
                logger.error(f"pdb2pqr30 reported success, but output file {output_pdb_file} is missing or empty.")
                logger.error(f"pdb2pqr30 stdout (when output file missing/empty):\n{completed_process.stdout}")
                logger.error(f"pdb2pqr30 stderr (when output file missing/empty):\n{completed_process.stderr}")
                return False
            return True

    except subprocess.TimeoutExpired as e:
        logger.error(f"pdb2pqr30 timed out for {pdb_file} after {e.timeout} seconds.")
        if e.stdout: # stdout/stderr might be bytes if timeout occurred very early
            logger.error(f"pdb2pqr30 stdout (on timeout):\n{e.stdout.decode(errors='replace') if isinstance(e.stdout, bytes) else e.stdout}")
        if e.stderr:
            logger.error(f"pdb2pqr30 stderr (on timeout):\n{e.stderr.decode(errors='replace') if isinstance(e.stderr, bytes) else e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"pdb2pqr30 command not found at {PDB2PQR_PATH}. "
                     "Please ensure PDB2PQR is installed and the path is correct.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running pdb2pqr30 for {pdb_file}: {e}")
        logger.error(traceback.format_exc())
        return False

# --- New NPZ Processing Functions ---

def decode_atom_name_from_4i1(encoded_name: np.ndarray) -> str:
    """
    Decodes a 4-integer array (presumably from ord(c) - 32) into a PDB atom name.
    """
    try:
        chars = []
        for val in encoded_name:
            if val == 0:  # Null terminator or padding
                break
            char = chr(int(val) + 32) # Assuming encoding was ord(c) - 32
            chars.append(char)
        name = "".join(chars).strip()

        # Format to 4 characters for PDB, attempting common conventions.
        if len(name) == 4: # e.g., "HD21"
            return name
        elif len(name) == 3: # e.g., "OXT", "HG1" (H on gamma C1)
            # If numeric is first, like "1HG", usually means H is first on G. PDB: "1HG "
            # If alpha is first, like "OXT", PDB: "OXT "
            return f"{name:<4}"
        elif len(name) == 2: # e.g., "CA", "SD"
            return f" {name:<2} " # Pad with space on left and right: " CA ", " SD "
        elif len(name) == 1: # e.g., "N", "C", "O"
            return f" {name}  "  # Pad with space on left and two on right: " N  "
        elif len(name) == 0:
             logger.warning(f"Decoded atom name from {encoded_name} is empty. Using fallback 'UNK'.")
             return "UNK " # Fallback, 4-chars
        else: # Longer than 4, truncate (should not happen with 4i1 if properly decoded)
            logger.warning(f"Decoded atom name '{name}' from {encoded_name} is longer than 4 chars. Truncating.")
            return name[:4]

    except Exception as e:
        logger.error(f"Error decoding atom name from {encoded_name}: {e}. Using fallback 'ERR'.")
        return "ERR " # Error fallback, 4-chars

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


# PDB_ATOM_NAME_FALLBACK_MAX_HEAVY = 5 # Number of initial heavy atoms to try mapping using simple order
# COMMON_RESIDUE_HEAVY_ATOM_ORDER = { # Simplified typical order for first few heavy atoms
#     0: 'N', 1: 'CA', 2: 'C', 3: 'O', 4: 'CB'
# }
# GLY_ATOM_ORDER = {0: 'N', 1: 'CA', 2: 'C', 3: 'O'} # Glycine lacks CB

# # Helper function for PDB atom name padding
# def _apply_pdb_atom_name_padding(name_stem: str) -> str:
#     name_len = len(name_stem)
#     if name_len == 1:  # Typically N, C, O, S etc.
#         return f" {name_stem}  "
#     elif name_len == 2:  # Typically CA, CB, CG, CD etc.
#         return f" {name_stem} "
#     elif name_len == 3:  # e.g. OXT, CG1, HD2
#         if name_stem == "OXT": # Special case for OXT
#             return " OXT"
#         # For other 3-char names, PDB standard varies (e.g., " CG1", "NE2 ").
#         # Simple left-alignment is used here as a compromise.
#         else:
#             return f"{name_stem:<4}"[:4]
#     elif name_len == 4:  # e.g. HD21 (ASN)
#         return name_stem
#     else: # Too short (should not happen if stems are valid) or too long
#         return f"{name_stem:<4}"[:4] # Truncate or pad


# STANDARD_ATOM_NOMENCLATURE = {
#     "ALA": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB"},
#     "ARG": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD", 7:"NE", 8:"CZ", 9:"NH1", 10:"NH2"},
#     "ASN": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"OD1", 7:"ND2"},
#     "ASP": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"OD1", 7:"OD2"},
#     "CYS": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"SG"},
#     "GLN": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD", 7:"OE1", 8:"NE2"},
#     "GLU": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD", 7:"OE1", 8:"OE2"},
#     "GLY": {0:"N", 1:"CA", 2:"C", 3:"O"},
#     "HIS": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"ND1", 7:"CD2", 8:"CE1", 9:"NE2"}, # Order based on common tautomer
#     "ILE": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG1", 6:"CD1", 7:"CG2"}, # CD1 is on CG1; CG2 on CB
#     "LEU": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD1", 7:"CD2"},
#     "LYS": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD", 7:"CE", 8:"NZ"},
#     "MET": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"SD", 7:"CE"},
#     "PHE": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD1", 7:"CD2", 8:"CE1", 9:"CE2", 10:"CZ"},
#     "PRO": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD"},
#     "SER": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"OG"},
#     "THR": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"OG1", 6:"CG2"},
#     "TRP": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD1", 7:"NE1", 8:"CE2", 9:"CD2", 10:"CE3", 11:"CZ2", 12:"CZ3", 13:"CH2"}, # Order can be complex
#     "TYR": {0:"N", 1:"CA", 2:"C", 3:"O", 4:"CB", 5:"CG", 6:"CD1", 7:"CD2", 8:"CE1", 9:"CE2", 10:"CZ", 11:"OH"},
# }


# def map_atom_info_to_pdb_atom_name(atom_npz_entry: np.ndarray,
#                                    residue_name: str,
#                                    atom_idx_in_residue_npz: int,
#                                    all_atom_entries_for_this_residue: list,
#                                    force_atom_name: str = None) -> str:
#     """
#     Heuristically maps atom information from NPZ to a PDB atom name,
#     unless force_atom_name is provided. Uses standard nomenclature where possible.
#     This is a simplified heuristic and might need significant refinement for accuracy.

#     Args:
#         atom_npz_entry: A single row from the NPZ 'atoms' array.
#                         Assumes atom_npz_entry[1] is atomic number.
#         residue_name (str): 3-letter code of the parent residue.
#         atom_idx_in_residue_npz (int): 0-based index of this atom within its residue's
#                                        list of atoms as read from NPZ 'atoms' array.
#         all_atom_entries_for_this_residue (list): List of all atom_npz_entry for the current residue.
#                                                   Used to count heavy atoms.

#     Returns:
#         str: A 4-character PDB atom name (e.g., " CA ", " CB ", " C1 ").
#     """
#     if force_atom_name:
#         return _apply_pdb_atom_name_padding(force_atom_name)

#     atomic_number = atom_npz_entry[1]
#     element_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, "X").upper() # Default to X if unknown

#     # Count how many heavy atoms appear before this one (or are this one) in the residue's NPZ list
#     heavy_atom_counter_for_this_atom = -1
#     current_atom_is_heavy = (atomic_number != 1) # Hydrogen is 1

#     for i, atm_entry in enumerate(all_atom_entries_for_this_residue):
#         if atm_entry[1] != 1: # if it's a heavy atom
#             heavy_atom_counter_for_this_atom +=1
#         if i == atom_idx_in_residue_npz: # Found the current atom
#             break

#     if not current_atom_is_heavy: # If it's a Hydrogen
#         # Basic hydrogen naming: H plus a number, or HN, HA etc. if logic was more complex
#         # PDB2PQR will rename these anyway. For now, generic.
#         # To make it somewhat unique for pdb2pqr, use its index within the residue.
#         # This formatting is specific and might not fit _apply_pdb_atom_name_padding well if name_stem is "H1", "H2" etc.
#         # Keeping its own formatting for now.
#         return f"{element_symbol:<2}{str(atom_idx_in_residue_npz + 1):<2}"[:4].ljust(4)

#     # --- Heavy Atom Naming ---

#     # 1. Try STANDARD_ATOM_NOMENCLATURE
#     if residue_name in STANDARD_ATOM_NOMENCLATURE:
#         atom_name_stem = STANDARD_ATOM_NOMENCLATURE[residue_name].get(heavy_atom_counter_for_this_atom)
#         if atom_name_stem:
#             return _apply_pdb_atom_name_padding(atom_name_stem)

#     # 2. Fallback to COMMON_RESIDUE_HEAVY_ATOM_ORDER (N, CA, C, O, CB)
#     # This is useful for non-standard residues or if STANDARD_ATOM_NOMENCLATURE is incomplete.
#     atom_order_map = GLY_ATOM_ORDER if residue_name == "GLY" else COMMON_RESIDUE_HEAVY_ATOM_ORDER
#     if heavy_atom_counter_for_this_atom < PDB_ATOM_NAME_FALLBACK_MAX_HEAVY: # Max heavy is 5 (0-4)
#         pdb_name_stem = atom_order_map.get(heavy_atom_counter_for_this_atom)
#         if pdb_name_stem:
#             # The old logic had specific padding based on element symbol vs pdb_name_stem,
#             # e.g. for 'C' in CA vs 'O' in O. _apply_pdb_atom_name_padding is simpler.
#             return _apply_pdb_atom_name_padding(pdb_name_stem)

#     # 3. Fallback: Generic element_symbol + count
#     # Use element symbol and a number based on its appearance order among heavy atoms in the residue
#     # e.g. C1, C2, N1 etc. This ensures uniqueness within the residue for PDB.
#     # PDB format: Element right justified if name is short (e.g. " C1 ", "S G ")
#     # Atom name: columns 13-16
#     # Element : columns 77-78
#     # For atom name, it's more like "CG1 ", "SD  "
#     # Let's try element + count relative to its element type in this residue

#     # Count occurrences of this element type up to this atom in the residue
#     element_type_count = 0
#     for i, atm_entry in enumerate(all_atom_entries_for_this_residue):
#         if atm_entry[1] == atomic_number: # Same element
#             element_type_count += 1
#         if i == atom_idx_in_residue_npz:
#             break

#     # Name like "C1", "C2", "N1" etc.
#     name_stem = f"{element_symbol}{element_type_count}" # e.g. C1, C2, N1
#     return _apply_pdb_atom_name_padding(name_stem)


def write_temp_pdb_from_npz(npz_data: dict, temp_pdb_path: str):
    """
    Writes a temporary PDB file from parsed NPZ data.
    Uses heuristic atom naming.

    New NPZ data structure assumptions for 'chains':
    - chain_entry[0]: chain ID (str, e.g., 'A')
    - chain_entry[5]: start index of this chain's residues in the global 'residues_data' array.
    - chain_entry[6]: number of residues in this chain.
    Other arrays ('atoms', 'residues') are assumed as before.
    """
    print(f"DEBUG: Entering write_temp_pdb_from_npz for temp file {temp_pdb_path}", flush=True)

    atoms_data = npz_data['atoms']
    residues_data = npz_data['residues']
    chains_data = npz_data['chains']

    atom_serial = 0
    atoms_written_count = 0

    with open(temp_pdb_path, 'w', encoding='utf-8') as f:
        # Iterate through chains as defined in chains_data
        for chain_idx_in_npz, chain_entry in enumerate(chains_data):
            chain_processed_atom_count = 0 # Ensure this is initialized for each chain
            if len(chain_entry) < 9: # Updated length check for indices 7 and 8
                logger.warning(f"Chain entry {chain_idx_in_npz} has too few fields (needs at least 9): {chain_entry}. Skipping chain.")
                continue

            chain_pdb_id_raw = str(chain_entry[0])
            chain_pdb_id = chain_pdb_id_raw.strip() if chain_pdb_id_raw.strip() else 'A' # Default if empty after strip
            if len(chain_pdb_id) > 1: chain_pdb_id = chain_pdb_id[0] # Take first char if multi-char

            res_start_idx_in_residues_array = int(chain_entry[7]) # Changed from 5 to 7
            num_residues_in_chain = int(chain_entry[8]) # Changed from 6 to 8
            print(f"DEBUG: Processing Chain ID: {chain_pdb_id}, NPZ chain_idx: {chain_idx_in_npz}, num_residues: {num_residues_in_chain}, res_start_idx: {res_start_idx_in_residues_array}", flush=True)

            # Iterate through residues in this chain
            print(f"DEBUG: Chain {chain_pdb_id} - num_residues_in_chain: {num_residues_in_chain}", flush=True)
            for res_offset_in_chain in range(num_residues_in_chain):
                res_idx_global_in_residues_array = res_start_idx_in_residues_array + res_offset_in_chain

                if res_idx_global_in_residues_array >= len(residues_data):
                    logger.warning(f"Residue index {res_idx_global_in_residues_array} out of bounds for residues_data (len {len(residues_data)}). Skipping residue for chain {chain_pdb_id}.")
                    continue

                res_entry_original = residues_data[res_idx_global_in_residues_array]

                # New interpretation based on request:
                # New interpretation based on request, including is_standard_residue:
                # [0]: res_name
                # [2]: res_seq_num_in_chain_0idx (for PDB numbering)
                # [3]: atom_start_global_idx
                # [4]: num_atoms_in_res_npz
                # [7]: is_standard_residue (boolean)
                if len(res_entry_original) < 8: # Need at least up to index 7 for these fields
                     logger.warning(f"Residue entry {res_idx_global_in_residues_array} has too few fields (needs at least 8): {res_entry_original}. Skipping residue for chain {chain_pdb_id}.")
                     continue

                res_name = str(res_entry_original[0])
                res_seq_num_for_pdb = int(res_entry_original[2]) + 1
                atom_start_global_idx = int(res_entry_original[3])
                num_atoms_in_res_npz = int(res_entry_original[4])
                is_standard_residue = bool(res_entry_original[7])
                record_type = "ATOM  " if is_standard_residue else "HETATM"

                # print(f"DEBUG:   Residue: {res_name}{res_seq_num_for_pdb} (Chain {chain_pdb_id}), NPZ res_glbl_idx: {res_idx_global_in_residues_array}, atom_start: {atom_start_global_idx}, num_atoms: {num_atoms_in_res_npz}", flush=True) # Commented out as per request

                all_atom_entries_for_this_residue_npz = atoms_data[atom_start_global_idx : atom_start_global_idx + num_atoms_in_res_npz]

                # carbonyl_o_assigned_in_residue = False # Removed for OXT logic reversion
                # is_c_terminal_residue = (res_offset_in_chain == num_residues_in_chain - 1) # Keep, general property
                # is_standard_protein_residue = is_standard_residue # Keep, general property (alias for is_standard_residue)


                # Iterate through atoms in this residue
                print(f"DEBUG:   Residue {res_name}{res_seq_num_for_pdb} (Chain {chain_pdb_id}) - num_atoms_in_res_npz: {num_atoms_in_res_npz}", flush=True)
                for atom_offset_in_residue in range(num_atoms_in_res_npz):
                    global_atom_idx_for_atoms_array = atom_start_global_idx + atom_offset_in_residue

                    if global_atom_idx_for_atoms_array >= len(atoms_data):
                        logger.warning(f"Global atom index {global_atom_idx_for_atoms_array} out of bounds for atoms_data (len {len(atoms_data)}). Skipping atom for residue {res_name}{res_seq_num_for_pdb}.")
                        continue

                    atom_npz_entry_original = atoms_data[global_atom_idx_for_atoms_array]
                    # print(f"DEBUG:     Atom global_idx: {global_atom_idx_for_atoms_array}, NPZ atom_entry: {atom_npz_entry_original!r}", flush=True) # Commented out as per request

                    if len(atom_npz_entry_original) < 4:
                        logger.warning(f"Atom {global_atom_idx_for_atoms_array} in file {npz_data.get('id', 'UNKNOWN_FILE')} has insufficient fields in 'atoms' array entry: {atom_npz_entry_original!r}. Skipping ATOM.")
                        continue

                    coord_list_from_atom_entry = atom_npz_entry_original[3]
                    if not hasattr(coord_list_from_atom_entry, '__getitem__') or not hasattr(coord_list_from_atom_entry, '__len__') or len(coord_list_from_atom_entry) < 3:
                        logger.warning(
                            f"Atom {global_atom_idx_for_atoms_array} in file {npz_data.get('id', 'UNKNOWN_FILE')} "
                            f"has unexpected coordinate structure in 'atoms' array entry[3]: {coord_list_from_atom_entry!r}. Skipping ATOM."
                        )
                        continue

                    x, y, z = coord_list_from_atom_entry[0], coord_list_from_atom_entry[1], coord_list_from_atom_entry[2]

                    # Reverted atom naming to use decode_atom_name_from_4i1
                    # Ensure atom_npz_entry_original[0] (encoded_name_field) is valid before decoding
                    # Assuming atom_npz_entry_original[0] is a numpy array like np.dtype("4i1")
                    if not hasattr(atom_npz_entry_original[0], '__iter__') or not atom_npz_entry_original[0].any():
                        logger.warning(f"Atom {global_atom_idx_for_atoms_array} in file {npz_data.get('id', 'UNKNOWN_FILE')} has invalid or empty encoded name field: {atom_npz_entry_original[0]!r}. Using fallback 'UNK'.")
                        atom_name_pdb = "UNK " # Default UNK with PDB padding for 3 chars + space
                    else:
                        encoded_name_field = atom_npz_entry_original[0]
                        atom_name_pdb = decode_atom_name_from_4i1(encoded_name_field)

                    current_atomic_number = atom_npz_entry_original[1] # Still needed for element symbol
                    element_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(current_atomic_number, 'X').rjust(2)

                    atom_serial += 1
                    chain_processed_atom_count += 1 # Increment per-chain atom counter

                    # Using res_name[:3] to ensure it's max 3 chars for PDB
                    # atom_name_pdb from decode_atom_name_from_4i1 should already be 4 chars.
                    pdb_line = (
                        f"{record_type}{atom_serial:5d} {atom_name_pdb}{res_name[:3]:<3s} {chain_pdb_id:1s}{res_seq_num_for_pdb:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element_symbol:<2s}\n"
                    )
                    f.write(pdb_line)
                    atoms_written_count += 1
                    if atoms_written_count % 500 == 0:
                        print(f"DEBUG:       ... {atoms_written_count} atoms written to PDB ...", flush=True)
                        logger.info(f"DEBUG: ... {atoms_written_count} atoms written to PDB for {npz_data.get('id', 'UNKNOWN_FILE')} ...")

            # After processing all residues in a chain, add TER record if any atoms were written for this chain
            if chain_processed_atom_count > 0:
                ter_serial = atom_serial + 1 # Serial for TER is last atom's serial + 1
                f.write(f"TER   {ter_serial:5d}      {res_name[:3]:<3s} {chain_pdb_id:1s}{res_seq_num_for_pdb:4d}\n")

        f.write("END\n") # Add newline to END record
    print(f"DEBUG: Exiting write_temp_pdb_from_npz for temp file {temp_pdb_path}, total atoms written: {atoms_written_count}", flush=True)
    if atoms_written_count == 0 and atom_serial == 0 :
        logger.warning(f"No atoms were written to temporary PDB file: {temp_pdb_path}. NPZ parsing or chain/residue iteration might have issues.")

# --- End of New NPZ Processing Functions ---


# --- Original functions get_atoms, generate_noesy_data remain unchanged ---
# ... (get_atoms and generate_noesy_data functions are here, unchanged from previous version) ...

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

    for npz_filename in npz_files: # Using npz_files directly if tqdm is not needed for now
        npz_file_path = os.path.join(args.input_dir, npz_filename)
        # base_name_npz = os.path.basename(npz_file_path) # Already got this
        name_part_npz, _ = os.path.splitext(npz_filename) # Use npz_filename for name_part

        logger.info(f"\nProcessing {npz_file_path}...")
        print(f"DEBUG: Top of loop for {npz_file_path}", flush=True)

        # Initialize names for finally block, and a list to gather files for removal
        temp_initial_pdb_name = None
        temp_hydro_pdb_name = None
        temp_files_to_remove = []

        try:
            print(f"DEBUG: Calling parse_npz for {npz_file_path}", flush=True)
            npz_data = parse_npz(npz_file_path)
            npz_data['id'] = Path(npz_file_path).stem # Add file ID for logging context
            print(f"DEBUG: Finished parse_npz for {npz_file_path}", flush=True)

            # Create temporary files using with statement for ensured creation before naming
            # delete=False means we are responsible for cleanup, done in finally block
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_initial.pdb', encoding='utf-8') as temp_initial_pdb_file_obj, \
                 tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_hydro.pdb', encoding='utf-8') as temp_hydro_pdb_file_obj:
                temp_initial_pdb_name = temp_initial_pdb_file_obj.name
                temp_hydro_pdb_name = temp_hydro_pdb_file_obj.name

            temp_files_to_remove.extend([temp_initial_pdb_name, temp_hydro_pdb_name])
            print(f"DEBUG: Temp files created: {temp_initial_pdb_name}, {temp_hydro_pdb_name}", flush=True)

            print(f"DEBUG: Calling write_temp_pdb_from_npz for {npz_file_path}", flush=True)
            write_temp_pdb_from_npz(npz_data, temp_initial_pdb_name)
            print(f"DEBUG: Finished write_temp_pdb_from_npz for {npz_file_path}", flush=True)

            # Save a copy of the initial PDB for debugging pdb2pqr
            if temp_initial_pdb_name and os.path.exists(temp_initial_pdb_name):
                debug_pdb_name = f"{name_part_npz}_debug_initial.pdb"
                debug_pdb_path = os.path.join(args.output_dir, debug_pdb_name)
                try:
                    shutil.copy2(temp_initial_pdb_name, debug_pdb_path)
                    logger.info(f"Saved debug PDB for pdb2pqr input to: {debug_pdb_path}")
                except Exception as e_copy:
                    logger.error(f"Could not copy debug PDB to {debug_pdb_path}: {e_copy}")

            if not os.path.exists(temp_initial_pdb_name) or os.path.getsize(temp_initial_pdb_name) == 0:
                logger.error(f"Initial PDB file {temp_initial_pdb_name} was not created or is empty after write_temp_pdb_from_npz. Skipping {npz_filename}.")
                continue # Goes to finally

            print(f"DEBUG: Calling add_hydrogens for {temp_initial_pdb_name}", flush=True)
            hydro_added_successfully = add_hydrogens(temp_initial_pdb_name, temp_hydro_pdb_name)
            print(f"DEBUG: Finished add_hydrogens for {temp_initial_pdb_name}, success: {hydro_added_successfully}", flush=True)

            if not hydro_added_successfully:
                logger.error(f"Skipping NOESY generation for {npz_file_path} due to hydrogen addition failure.")
                continue # Goes to finally

            logger.info(f"Successfully added hydrogens: {temp_hydro_pdb_name}")

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
            # Clean up temporary files explicitly added to the list
            for temp_file_path_to_remove in temp_files_to_remove:
                if temp_file_path_to_remove and os.path.exists(temp_file_path_to_remove):
                    try:
                        os.remove(temp_file_path_to_remove)
                        logger.debug(f"Removed temp file: {temp_file_path_to_remove}")
                    except OSError as e:
                        logger.error(f"Error removing temp file {temp_file_path_to_remove}: {e}")

    logger.info("\nProcessing complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
