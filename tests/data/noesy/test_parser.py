import unittest
import tempfile
import os
import shutil
import re # For the parser's _parse_residue_id method testing (indirectly)

from boltz.data.noesy.parser import NOESYParser

class TestNOESYParser(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to store test files
        self.test_dir = tempfile.mkdtemp()

        # Sample NOESY file content
        self.valid_noesy_content = """
# This is a comment line
A1 B5 1 2.5 HA HN
C10 D12 2 3.0 HB CA
A1 B5 3 2.8 HX HY ; Another peak for A1-B5
A101 GLY105 4 1.5 HA N
# Ambiguous peak example
ALA10 LYS20 5 4.0 N H
ALA10 ARG30 5 4.2 N H
"""
        self.mixed_noesy_content = """
VALID_RES1 VALID_RES2 1 2.0 A B
INVALID_LINE_TOO_FEW_PARTS
ALA10 LYS20 2 3.5 C D
X1 Y2 3 BAD_DIST E F
GLY5 HIS6 4 4.0 G H I J K # Too many parts
PRO7 SER8 5 1.0 L M
"""
        self.empty_noesy_content = ""
        self.format_issue_content = "RES1 RES2 PEAK_ID DIST ATOM1 ATOM2\n" # Header like, but parser expects data

        self.path_valid = os.path.join(self.test_dir, "valid.txt")
        self.path_mixed = os.path.join(self.test_dir, "mixed.txt")
        self.path_empty = os.path.join(self.test_dir, "empty.txt")
        self.path_format_issue = os.path.join(self.test_dir, "format_issue.txt")
        self.path_non_existent = os.path.join(self.test_dir, "non_existent.txt")

        with open(self.path_valid, "w") as f:
            f.write(self.valid_noesy_content)
        with open(self.path_mixed, "w") as f:
            f.write(self.mixed_noesy_content)
        with open(self.path_empty, "w") as f:
            f.write(self.empty_noesy_content)
        with open(self.path_format_issue, "w") as f:
            f.write(self.format_issue_content)

        self.parser = NOESYParser()

    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)

    def test_parse_valid_file(self):
        parsed_data = self.parser.parse(self.path_valid)

        self.assertIn('peaks', parsed_data)
        self.assertIn('grouped_by_peak_id', parsed_data)

        peaks = parsed_data['peaks']
        grouped = parsed_data['grouped_by_peak_id']

        # Expected number of lines (6 actual data lines)
        self.assertEqual(len(peaks), 6)

        # Check first peak (A1 B5 1 2.5 HA HN)
        # Residue IDs are 1-based in file, 0-based in parser. 'A1' -> 0, 'B5' -> 4
        # The parser's _parse_residue_id extracts numeric part.
        # 'A1' -> 0, 'B5' -> 4 (assuming B is not part of number)
        # 'C10' -> 9, 'D12' -> 11
        # 'A101' -> 100, 'GLY105' -> 104
        # 'ALA10' -> 9, 'LYS20' -> 19, 'ARG30' -> 29

        peak1 = next(p for p in peaks if p['peak_id'] == 1 and p['res_from_full'] == 'A1')
        self.assertEqual(peak1['res_from'], 0) # A1 -> 0
        self.assertEqual(peak1['res_to'], 4)   # B5 -> 4
        self.assertEqual(peak1['distance'], 2.5)
        self.assertEqual(peak1['atom_from'], "HA")
        self.assertEqual(peak1['atom_to'], "HN")

        peak4 = next(p for p in peaks if p['peak_id'] == 4)
        self.assertEqual(peak4['res_from'], 100) # A101 -> 100
        self.assertEqual(peak4['res_to'], 104)  # GLY105 -> 104
        self.assertEqual(peak4['distance'], 1.5)

        # Check ambiguity grouping for peak_id 5
        self.assertIn(5, grouped)
        self.assertEqual(len(grouped[5]), 2)
        peak5_assignments = grouped[5]

        res_from_values_peak5 = {p['res_from'] for p in peak5_assignments} # Should be ALA10 -> 9
        self.assertEqual(res_from_values_peak5, {9})

        # Check that both assignments for peak 5 are present in the main 'peaks' list
        peak5_lys_found = any(p['peak_id'] == 5 and p['res_to_full'] == 'LYS20' for p in peaks)
        peak5_arg_found = any(p['peak_id'] == 5 and p['res_to_full'] == 'ARG30' for p in peaks)
        self.assertTrue(peak5_lys_found)
        self.assertTrue(peak5_arg_found)


    def test_parse_mixed_file(self):
        # mixed_noesy_content has 2 valid lines, 3 invalid lines
        # VALID_RES1 VALID_RES2 1 2.0 A B
        # ALA10 LYS20 2 3.5 C D
        # PRO7 SER8 5 1.0 L M
        parsed_data = self.parser.parse(self.path_mixed)
        peaks = parsed_data['peaks']

        # Expect 3 valid peaks after skipping errors
        self.assertEqual(len(peaks), 3)

        # Check one of the valid peaks
        # Example: PRO7 SER8 5 1.0 L M -> PRO7 (6), SER8 (7)
        peak_pro_ser = next(p for p in peaks if p['peak_id'] == 5)
        self.assertEqual(peak_pro_ser['res_from'], 6) # PRO7 -> 6
        self.assertEqual(peak_pro_ser['res_to'], 7)   # SER8 -> 7
        self.assertEqual(peak_pro_ser['distance'], 1.0)


    def test_parse_empty_file(self):
        parsed_data = self.parser.parse(self.path_empty)
        self.assertEqual(len(parsed_data['peaks']), 0)
        self.assertEqual(len(parsed_data['grouped_by_peak_id']), 0)

    def test_parse_format_issue_file(self):
        # This file only contains a header-like line, which will be skipped due to conversion errors.
        parsed_data = self.parser.parse(self.path_format_issue)
        self.assertEqual(len(parsed_data['peaks']), 0)

    def test_parse_non_existent_file(self):
        parsed_data = self.parser.parse(self.path_non_existent)
        self.assertEqual(len(parsed_data['peaks']), 0)
        self.assertEqual(len(parsed_data['grouped_by_peak_id']), 0)
        # Optionally, check for logged error if logging was mocked/captured

    def test_parse_residue_id_internal_method(self):
        # Test the internal helper directly, though it's also tested via `parse`
        self.assertEqual(self.parser._parse_residue_id("A1"), 0)
        self.assertEqual(self.parser._parse_residue_id("ARG101"), 100)
        self.assertEqual(self.parser._parse_residue_id("10"), 9)
        with self.assertRaises(ValueError):
            self.parser._parse_residue_id("ABC") # No digits
        with self.assertRaises(ValueError):
            self.parser._parse_residue_id("")


if __name__ == '__main__':
    unittest.main()
