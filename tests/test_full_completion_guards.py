from pathlib import Path
import tempfile
import unittest
from stg_ddqn.full_audit import audit

ROOT=Path(__file__).resolve().parents[1]


class FullCompletionGuards(unittest.TestCase):
    def test_short_validation_cannot_pass_full_duration_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'600/3600'):
                audit(ROOT/'configs/sumo_validation.yaml',directory,directory)

    def test_incomplete_full_study_cannot_publish_final_archive(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'delivery'
            result=subprocess.run([sys.executable,str(ROOT/'tools/package_full_delivery.py'),
                '--results',directory,'--delivery',str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('unfinished full study',result.stderr)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
