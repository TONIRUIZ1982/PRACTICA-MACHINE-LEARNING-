"""Tests for the non-invasive delivery checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.check_delivery import existing_required_files


class DeliveryChecksTests(unittest.TestCase):
    def test_reports_missing_required_files(self) -> None:
        with TemporaryDirectory() as directory:
            missing = existing_required_files(Path(directory))

        self.assertIn("docs/VARIABLES.md", missing)
        self.assertIn("docs/GUIA_DEFENSA.md", missing)


if __name__ == "__main__":
    unittest.main()
