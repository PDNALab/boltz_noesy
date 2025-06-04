import argparse
import os
import random
import subprocess
import tempfile

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.vectors import Vector
import numpy as np # Will be used for distance calculation

# CASP13 naming conventions for relevant atoms
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

# Backbone amide hydrogen
BACKBONE_AMIDE = "H" # Sometimes "HN" but PDB2PQR usually names it 'H' if attached to 'N'

PDB2PQR_PATH = "/home/swebot/.local/bin/pdb2pqr30"

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
    Main function to orchestrate the NOESY data generation process.
    """
    parser = argparse.ArgumentParser(description="Generate NOESY-like data from PDB/mmCIF files.")
    parser.add_argument("input_dir", help="Directory containing input PDB/mmCIF files.")
    parser.add_argument("output_dir", help="Directory to save generated NOESY data.")
    parser.add_argument("--distance_cutoff", type=float, default=5.0, help="Distance cutoff for NOESY contacts (Angstroms).")
    # Add more arguments if needed, e.g., for PDB2PQR options

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_files = []
    for filename in os.listdir(args.input_dir):
        # Process .pdb and .cif (as PDB2PQR might handle some .cif or we might add conversion)
        if filename.lower().endswith(".pdb") or filename.lower().endswith(".cif") or filename.lower().endswith(".ent"):
            input_files.append(os.path.join(args.input_dir, filename))

    if not input_files:
        print(f"No PDB or mmCIF files found in {args.input_dir}")
        return

    for input_file_path in input_files:
        base_name = os.path.basename(input_file_path)
        name_part, ext = os.path.splitext(base_name)

        print(f"\nProcessing {input_file_path}...")

        # Use a temporary file for the PDB with hydrogens.
        # This file will be deleted after processing.
        temp_pdb_h_fd, pdb_with_h_path = tempfile.mkstemp(suffix=".pdb")
        os.close(temp_pdb_h_fd) # Close file descriptor, PDB2PQR will open by path

        try:
            # PDB2PQR typically requires PDB input.
            # If the input is mmCIF, it ideally should be converted to PDB first.
            # For this script, we'll pass the path directly. Some PDB2PQR versions
            # might handle mmCIF or the user might provide PDBs.
            # A robust solution would involve:
            # if input_file_path.lower().endswith(".cif"):
            #    print(f"Converting CIF {input_file_path} to temporary PDB for PDB2PQR...")
            #    cif_parser = MMCIFParser()
            #    structure = cif_parser.get_structure("temp_cif", input_file_path)
            #    pdb_io = Bio.PDB.PDBIO()
            #    pdb_io.set_structure(structure)
            #    temp_pdb_for_h_add_fd, temp_pdb_path = tempfile.mkstemp(suffix=".pdb")
            #    os.close(temp_pdb_for_h_add_fd)
            #    pdb_io.save(temp_pdb_path)
            #    current_input_for_hydrogenation = temp_pdb_path
            #    file_to_clean_additionally = temp_pdb_path
            # else:
            current_input_for_hydrogenation = input_file_path
            file_to_clean_additionally = None

            add_hydrogens(current_input_for_hydrogenation, pdb_with_h_path)

            if not os.path.exists(pdb_with_h_path) or os.path.getsize(pdb_with_h_path) == 0:
                print(f"Hydrogen addition failed or produced an empty file for {input_file_path}, skipping.")
                continue # Skip to the next file

            # Generate NOESY data using the hydrogenated PDB file
            # The generate_noesy_data function uses PDBParser, so it expects a PDB file.
            noesy_data = generate_noesy_data(pdb_with_h_path, args.distance_cutoff)

            if noesy_data:
                output_noesy_filename = os.path.join(args.output_dir, f"{name_part}_noesy.txt")
                with open(output_noesy_filename, "w") as f_out:
                    for line in noesy_data:
                        f_out.write(line + "\n")
                print(f"Generated NOESY data for {base_name} at {output_noesy_filename}")
            else:
                print(f"No NOESY data generated for {base_name}.")

        except Exception as e:
            print(f"Error processing file {input_file_path}: {e}")
        finally:
            # Clean up temporary PDB file with hydrogens
            if os.path.exists(pdb_with_h_path):
                os.remove(pdb_with_h_path)
            # if file_to_clean_additionally and os.path.exists(file_to_clean_additionally):
            #    os.remove(file_to_clean_additionally)


    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
