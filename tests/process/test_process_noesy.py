import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import subprocess
import tempfile
import os
import argparse
import io
import numpy as np
import logging
import itertools # For compute_contacts_new_method
import shutil # For main test cleanup

# Functions and constants to be tested
from scripts.process.process_noesy import (
    add_hydrogens,
    extract_filtered_protons,
    compute_contacts_new_method,
    main as process_noesy_main,
    parse_npz,
    decode_atom_name_from_4i1,
    write_temp_pdb_from_npz,
    PDB2PQR_PATH,
    ATOMIC_NUMBER_TO_SYMBOL,
    H_MATCH_TOLERANCE,
    # NEW_DISTANCE_CUTOFF, # This is a default in argparse, not used directly by compute_contacts
    DISTANCE_NOE_THRESHOLD,
    NOISE_STD_H_SHIFT_SIM,
    TARGET_HYDROPHOBIC_RESIDUES,
    simulate_shift
)

from Bio.PDB import PDBParser, Structure, Model, Chain, Residue
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.vectors import Vector

# Helper to encode atom names for mock NPZ data
def encode_atom_name_to_4i1(core_name: str) -> np.ndarray:
    encoded = np.zeros(4, dtype=np.int8)
    for i, char_val in enumerate(core_name.strip()[:4]):
        if i < 4: encoded[i] = ord(char_val) - 32
        else: break
    return encoded

# Helper to create Bio.PDB.Atom for test structures
def create_test_atom(name, element, coord_tuple):
    atom = BioAtom(name, Vector(coord_tuple), 0, 0, None, name, 0, element.upper())
    return atom

# Helper to create Bio.PDB.Residue for test structures
def create_test_residue(resname: str, res_seq_num: int, atoms_list: list, chain_id: str = 'A') -> Residue:
    hetfield = ' ' if resname in TARGET_HYDROPHOBIC_RESIDUES or resname == "GLY" or resname == "ALA" else 'H_'+resname
    res_id = (hetfield, res_seq_num, ' ')
    residue = Residue(res_id, resname, '    ')
    for atom in atoms_list:
        residue.add(atom)
    return residue

# Helper to create Bio.PDB.Structure for test structures
def create_test_structure(chain_residue_map: dict) -> Structure:
    structure = Structure("test_structure")
    model = Model(0)
    for chain_id, residue_list in chain_residue_map.items():
        chain = Chain(chain_id)
        for residue in residue_list:
            chain.add(residue)
        model.add(chain)
    structure.add(model)
    return structure


class TestProcessNoesyParts(unittest.TestCase): # Renamed for clarity
    def setUp(self):
        self.maxDiff = None
        self.dummy_input_pdb = "dummy_input.pdb"
        self.dummy_output_pdb = "dummy_output.pdb"

        # More comprehensive NPZ data for write_temp_pdb_from_npz
        self.mock_npz_data_detailed = {
            'atoms': np.array([
                # Atom: encoded_name, atomic_num, ?, coords, ?, is_hetatm_equivalent_false, ?
                (encode_atom_name_to_4i1("N"),   7, 0, [1.0, 2.0, 3.0], [0]*3, False, 0), # ALA 1 Atom 0
                (encode_atom_name_to_4i1("CA"),  6, 0, [1.5, 2.5, 3.5], [0]*3, False, 0), # ALA 1 Atom 1
                (encode_atom_name_to_4i1("C"),   6, 0, [0.0, 0.0, 0.0], [0]*3, False, 0), # ALA 1 Atom 2 (Zero Coords)
                (encode_atom_name_to_4i1("O"),   8, 0, [2.0, 3.0, 4.0], [0]*3, False, 0), # ALA 1 Atom 3
                (encode_atom_name_to_4i1("N"),   7, 0, [10.0,12.0,13.0],[0]*3, False, 0), # GLY 2 Atom 0
                (encode_atom_name_to_4i1("CA"),  6, 0, [10.5,12.5,13.5],[0]*3, False, 0), # GLY 2 Atom 1
                (encode_atom_name_to_4i1("P"),  15,0, [20.0,22.0,23.0],[0]*3, False, 0), # LIG 3 Atom 0 (Non-standard)
            ], dtype=object),
            'coords': np.array([ # Not directly used by write_temp_pdb if atoms_data[i][3] has coords
                ([1.0,2.0,3.0],), ([1.5,2.5,3.5],), ([0.0,0.0,0.0],), ([2.0,3.0,4.0],),
                ([10.0,12.0,13.0],), ([10.5,12.5,13.5],), ([20.0,22.0,23.0],)
            ], dtype=object),
            'residues': np.array([
                # Res: name, ?, res_seq_num_0_idx, atom_start_idx, num_atoms, ?, ?, is_standard_true
                ('ALA', 0, 0, 0, 4, None, None, True),  # Res 0 (ALA, seq 1), 4 atoms starting at 0
                ('GLY', 0, 1, 4, 2, None, None, True),  # Res 1 (GLY, seq 2), 2 atoms starting at 4
                ('LIG', 0, 2, 6, 1, None, None, False), # Res 2 (LIG, seq 3), 1 atom starting at 6 (Non-standard)
            ], dtype=object),
            'chains': np.array([
                # Chain: pdb_id, ?, ?, ?, ?, ?, ?, res_start_idx, num_res
                ('A', None,None,None,None,None,None, 0, 3) # Chain A, 3 residues starting at 0
            ], dtype=object)
        }

    def test_decode_atom_name_formatting(self):
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("N")),   " N  ")
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("CA")),  " CA ")
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("OXT")), " OXT")
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("OH")),  " OH ")
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("CG1")), "CG1 ")
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("HD21")),"HD21")
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("1H")),  "1H  ")
        self.assertEqual(decode_atom_name_from_4i1(encode_atom_name_to_4i1("2HG")), "2HG ")
        empty_encoded = np.zeros(4, dtype=np.int8)
        self.assertEqual(decode_atom_name_from_4i1(empty_encoded), "UNK ")

    @patch('numpy.load')
    def test_parse_npz_success(self, mock_np_load):
        mock_np_load.return_value = {
            'atoms': np.array([]), 'coords': np.array([]),
            'residues': np.array([]), 'chains': np.array([])
        }
        data = parse_npz("dummy.npz")
        self.assertIn('atoms', data)
        self.assertIn('coords', data)
        self.assertIn('residues', data)
        self.assertIn('chains', data)
        mock_np_load.assert_called_once_with("dummy.npz")

    @patch('numpy.load')
    def test_parse_npz_file_not_found(self, mock_np_load):
        mock_np_load.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            parse_npz("nonexistent.npz")

    @patch('numpy.load')
    def test_parse_npz_key_error(self, mock_np_load):
        mock_np_load.return_value = {'atoms': np.array([])} # Missing keys
        with self.assertRaises(KeyError):
            parse_npz("badformat.npz")

    @patch('scripts.process.process_noesy.logger')
    def test_write_temp_pdb_from_npz(self, mock_logger):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".pdb") as tmpfile:
            temp_pdb_path = tmpfile.name

        write_temp_pdb_from_npz(self.mock_npz_data_detailed, temp_pdb_path)

        with open(temp_pdb_path, 'r') as f:
            pdb_content = f.read()

        # Atom serials are 1-based
        # Res seq are 1-based (0-idx from npz + 1)
        expected_pdb_content = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C
ATOM      3  O   ALA A   1       2.000   3.000   4.000  1.00  0.00           O
TER       4      ALA A   1
ATOM      5  N   GLY A   2      10.000  12.000  13.000  1.00  0.00           N
ATOM      6  CA  GLY A   2      10.500  12.500  13.500  1.00  0.00           C
TER       7      GLY A   2
END
"""
        # Using splitlines and comparing line by line can be more robust to minor whitespace diffs if any
        self.assertEqual(pdb_content.splitlines(), expected_pdb_content.splitlines())

        # Check for logging of skipped zero-coord atom
        zero_coord_log_found = False
        for call_args in mock_logger.info.call_args_list:
            if "NPZ idx:2 (0,0,0) coords. Skip." in call_args[0][0]:
                zero_coord_log_found = True
                break
        self.assertTrue(zero_coord_log_found, "Log message for zero-coordinate atom not found.")

        # Check logging for non-standard residue (LIG) - it should be skipped, so no atoms from it.
        # (write_temp_pdb_from_npz doesn't log skipped standard residues, only warns for data issues)

        os.remove(temp_pdb_path)

    @patch('numpy.random.normal')
    def test_simulate_shift(self, mock_np_random_normal):
        mock_np_random_normal.return_value = 0.0
        self.assertEqual(simulate_shift(np.array([0.0,0.0,0.0])), 0.000)
        self.assertEqual(simulate_shift(np.array([10.0,0.0,0.0])), 1.000) # norm is 10, 10*0.1=1.0
        mock_np_random_normal.return_value = 0.0053
        self.assertEqual(simulate_shift(np.array([3.0,4.0,0.0])), 0.505) # norm is 5, 5*0.1=0.5, 0.5+0.0053 = 0.5053 -> 0.505

    def test_extract_filtered_protons(self):
        ala_res_A1 = create_test_residue("ALA", 1, [
            create_test_atom(" N  ", "N", (1,0,0)), create_test_atom(" H  ", "H", (1,1,0)),
            create_test_atom(" HB1", "H", (1,2,0))
        ])
        ser_res_A2 = create_test_residue("SER", 2, [ # Not in TARGET_HYDROPHOBIC_RESIDUES
            create_test_atom(" N  ", "N", (2,0,0)), create_test_atom(" H  ", "H", (2,1,0))
        ])
        val_res_B1 = create_test_residue("VAL", 1, [
            create_test_atom(" N  ", "N", (3,0,0)), create_test_atom("HG11", "H", (3,1,0)),
            create_test_atom(" CA ", "C", (3,2,0)) # Non-proton
        ])
        lig_atoms = [create_test_atom(" H1 ", "H", (4,0,0))] # Non-standard residue
        lig_res_B2 = Residue(('H_LIG', 2, ' '), "LIG", "    "); [lig_res_B2.add(a) for a in lig_atoms]

        mock_structure = create_test_structure({'A': [ala_res_A1, ser_res_A2], 'B': [val_res_B1, lig_res_B2]})
        protons = extract_filtered_protons(mock_structure)

        self.assertEqual(len(protons), 3)
        proton_info = sorted([(p['chain_id'], p['res_num'], p['atom_name']) for p in protons])
        self.assertIn(('A', 1, 'H'), proton_info)
        self.assertIn(('A', 1, 'HB1'), proton_info)
        self.assertIn(('B', 1, 'HG11'), proton_info)

    @patch('scripts.process.process_noesy.simulate_shift')
    def test_compute_contacts_new_method(self, mock_simulate_shift):
        protons_list = [
            {'chain_id': 'A', 'res_num': 10, 'atom_name': 'HA',  'coord': np.array([0.0,0.0,0.0])},
            {'chain_id': 'A', 'res_num': 12, 'atom_name': 'HB1', 'coord': np.array([0.0,0.0,3.0])}, # dist 3.0
            {'chain_id': 'A', 'res_num': 14, 'atom_name': 'HG2', 'coord': np.array([0.0,0.0,6.0])}, # dist 6.0
            {'chain_id': 'A', 'res_num': 16, 'atom_name': 'HD1', 'coord': np.array([0.0,0.0,8.0])}, # dist 8.0
            {'chain_id': 'A', 'res_num': 18, 'atom_name': 'HE1', 'coord': np.array([0.0,0.0,3.5])}, # Shift diff test
        ]
        for p in protons_list: p['atom_obj'] = None

        def shift_side_effect(coord):
            if np.array_equal(coord, protons_list[0]['coord']): return 1.00 # p0 (A10 HA)
            if np.array_equal(coord, protons_list[1]['coord']): return 1.01 # p1 (A12 HB1) -> |1.00-1.01|=0.01 <= H_MATCH_TOLERANCE
            if np.array_equal(coord, protons_list[2]['coord']): return 1.02 # p2 (A14 HG2) -> |1.00-1.02|=0.02 <= H_MATCH_TOLERANCE
            if np.array_equal(coord, protons_list[4]['coord']): return 2.00 # p4 (A18 HE1) -> |1.00-2.00|=1.00 > H_MATCH_TOLERANCE
            return 0.0
        mock_simulate_shift.side_effect = shift_side_effect

        # Using specific cutoffs for this test, not global defaults directly
        # initial_distance_cutoff affects which pairs are considered AT ALL
        # actual_noe_distance_threshold affects the peak_type (1 or 0)
        contacts = compute_contacts_new_method(protons_list,
                                               initial_distance_cutoff=7.5,
                                               actual_noe_distance_threshold=5.0) # Standard NOE is <= 5A

        self.assertEqual(len(contacts), 2) # (0,1) and (0,2) should pass. (0,3) dist 8.0 > 7.5. (0,4) shift diff too large.

        contact_0_1 = next(c for c in contacts if c['res1_num']==10 and c['atom1_name']=='HA' and c['res2_num']==12 and c['atom2_name']=='HB1')
        self.assertAlmostEqual(contact_0_1['distance'], 3.00)
        self.assertEqual(contact_0_1['peak_type'], 1) # 3.0 <= 5.0 (actual_noe_threshold)

        contact_0_2 = next(c for c in contacts if c['res1_num']==10 and c['atom1_name']=='HA' and c['res2_num']==14 and c['atom2_name']=='HG2')
        self.assertAlmostEqual(contact_0_2['distance'], 6.00)
        self.assertEqual(contact_0_2['peak_type'], 0) # 6.0 > 5.0 (actual_noe_threshold)

    @patch('os.remove')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_add_hydrogens_success(self, mock_os_getsize, mock_os_exists, mock_subprocess_run, mock_os_remove):
        mock_cp = MagicMock(spec=subprocess.CompletedProcess, returncode=0)
        mock_subprocess_run.return_value = mock_cp
        expected_dummy_pqr = self.dummy_output_pdb + ".pqr_dummy"

        # Simulate pdb2pqr creating the output and dummy files
        def os_exists_side_effect(path_arg):
            if path_arg == self.dummy_output_pdb: return True # Output PDB exists
            if path_arg == expected_dummy_pqr: return True # Dummy PQR exists initially
            return False # Other paths
        mock_os_exists.side_effect = os_exists_side_effect
        mock_os_getsize.return_value = 100 # Output PDB is not empty

        result = add_hydrogens(self.dummy_input_pdb, self.dummy_output_pdb)
        self.assertTrue(result)
        expected_cmd = [ PDB2PQR_PATH, "--ff=AMBER", "--pdb-output", self.dummy_output_pdb, self.dummy_input_pdb, expected_dummy_pqr ]
        mock_subprocess_run.assert_called_once_with(expected_cmd, capture_output=True, text=True, timeout=600, check=False)
        mock_os_remove.assert_any_call(expected_dummy_pqr) # Ensure cleanup is attempted


class TestProcessNoesyMain(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.test_input_dir = tempfile.mkdtemp(prefix="test_input_")
        self.test_output_dir = tempfile.mkdtemp(prefix="test_output_")
        self.npz_filename = "test_protein.npz"
        self.npz_file_path = os.path.join(self.test_input_dir, self.npz_filename)

        # Create a minimal valid NPZ file for the happy path
        self.mock_npz_content = {
            'atoms': np.array([
                (encode_atom_name_to_4i1("N"), 7,0,[1,2,3],[0]*3,False,0),
                (encode_atom_name_to_4i1("CA"),6,0,[2,3,4],[0]*3,False,0),
                (encode_atom_name_to_4i1("H"), 1,0,[1,1,2],[0]*3,False,0), # Proton for ALA
                (encode_atom_name_to_4i1("N"), 7,0,[10,12,13],[0]*3,False,0),
                (encode_atom_name_to_4i1("CA"),6,0,[11,13,14],[0]*3,False,0),
                (encode_atom_name_to_4i1("HB1"),1,0,[10,11,12],[0]*3,False,0) # Proton for VAL (res 2)
            ], dtype=object),
            'residues': np.array([
                ('ALA', 0, 0, 0, 3, None, None, True), # res_seq_num 1 (0-idx), 3 atoms
                ('VAL', 0, 1, 3, 3, None, None, True)  # res_seq_num 2 (1-idx), 3 atoms
            ], dtype=object),
            'chains': np.array([('A',None,None,None,None,None,None,0,2)], dtype=object)
        }
        np.savez(self.npz_file_path, **self.mock_npz_content)

    def tearDown(self):
        shutil.rmtree(self.test_input_dir)
        shutil.rmtree(self.test_output_dir)

    @patch('scripts.process.process_noesy.add_hydrogens')
    @patch('Bio.PDB.PDBParser.get_structure')
    @patch('scripts.process.process_noesy.extract_filtered_protons')
    @patch('scripts.process.process_noesy.compute_contacts_new_method')
    @patch('scripts.process.process_noesy.logger') # To check log messages
    def test_main_happy_path(self, mock_logger, mock_compute_contacts, mock_extract_protons, mock_get_structure, mock_add_hydrogens):
        # --- Mocks Setup ---
        mock_add_hydrogens.return_value = True # Simulate successful hydrogen addition

        # Mock PDB structure
        mock_atom_ala_h = create_test_atom(" H  ", "H", (1,1,2))
        mock_atom_val_hb1 = create_test_atom(" HB1", "H", (10,11,12))
        mock_residue_ala = create_test_residue("ALA", 1, [mock_atom_ala_h])
        mock_residue_val = create_test_residue("VAL", 2, [mock_atom_val_hb1]) # VAL is target
        mock_structure_obj = create_test_structure({'A': [mock_residue_ala, mock_residue_val]})
        mock_get_structure.return_value = mock_structure_obj

        # Mock proton extraction
        protons_extracted = [
            {'atom_obj': mock_atom_ala_h, 'coord': mock_atom_ala_h.coord, 'res_num': 1, 'chain_id': 'A', 'atom_name': 'H'},
            {'atom_obj': mock_atom_val_hb1, 'coord': mock_atom_val_hb1.coord, 'res_num': 2, 'chain_id': 'A', 'atom_name': 'HB1'}
        ]
        mock_extract_protons.return_value = protons_extracted

        # Mock contact computation
        contacts_computed = [
            {'chain_id': 'A', 'res1_num': 1, 'atom1_name': 'H',
             'res2_num': 2, 'atom2_name': 'HB1',
             'distance': 3.50, 'peak_type': 1}
        ]
        mock_compute_contacts.return_value = contacts_computed

        # --- Run main ---
        test_args = argparse.Namespace(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            distance_cutoff=7.5 # This is the default, passed to compute_contacts
        )
        with patch('argparse.ArgumentParser.parse_args', return_value=test_args):
            process_noesy_main()

        # --- Assertions ---
        # Check if add_hydrogens was called (it creates temp files, names will be dynamic)
        mock_add_hydrogens.assert_called_once()
        self.assertTrue(mock_add_hydrogens.call_args[0][0].endswith("_initial.pdb")) # input to add_hydrogens
        self.assertTrue(mock_add_hydrogens.call_args[0][1].endswith("_hydro.pdb"))   # output of add_hydrogens

        mock_get_structure.assert_called_once() # With the hydro PDB
        self.assertTrue(mock_get_structure.call_args[0][1].endswith("_hydro.pdb"))

        mock_extract_protons.assert_called_once_with(mock_structure_obj)

        mock_compute_contacts.assert_called_once_with(
            protons_extracted,
            initial_distance_cutoff=7.5, # From args
            actual_noe_distance_threshold=DISTANCE_NOE_THRESHOLD # Global constant
        )

        # Check output file
        expected_output_filename = os.path.join(self.test_output_dir, f"{self.npz_filename.replace('.npz', '.txt')}")
        self.assertTrue(os.path.exists(expected_output_filename))
        with open(expected_output_filename, 'r') as f:
            output_content = f.read()

        expected_header = "ChainID\tRes1_Num\tRes2_Num\tPeak_Type\tDistance\tAtom1_Name\tAtom2_Name\n"
        expected_data_line = "A\t1\t2\t1\t3.50\tH\tHB1\n"
        self.assertEqual(output_content, expected_header + expected_data_line)

        # Check debug PDB copy
        expected_debug_pdb = os.path.join(self.test_output_dir, "test_protein_debug_initial.pdb")
        self.assertTrue(os.path.exists(expected_debug_pdb))

        # Check for relevant log messages
        mock_logger.info.assert_any_call(f"Generated 1 contacts for {self.npz_filename.replace('.npz', '')} at {expected_output_filename}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING) # Keep tests quieter unless debugging
    unittest.main(verbosity=2)

```
