"""Read-only school-machine setup verification; no design cases needed yet."""
from pathlib import Path
import sys
RC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RC))
from Data_Generation import Run_Weighted_Seismic_Calibration as runner

def main():
    saved = runner.read_json(Path(__file__).with_name('school_setup_identity.json'))
    failures = []
    for name, actual in [('sources', runner.source_identity()), ('runtime', runner.runtime_identity())]:
        if saved[name] != actual:
            failures.append(name + ' differs from the prepared school package')
    for path, sha in saved['input_sha256'].items():
        file = RC / path
        if not file.is_file() or runner.digest(file) != sha:
            failures.append('missing/changed record or metadata: ' + path)
    for path, sha in saved['launcher_sha256'].items():
        file = RC / path
        if not file.is_file() or runner.digest(file) != sha:
            failures.append('missing/changed launcher: ' + path)
    if failures:
        raise SystemExit('SETUP MISMATCH:\n  ' + '\n  '.join(failures))
    print(f"PASS: prepared source, runtime, launchers and {saved['record_pair_count']} paired ground motions match. No analyses launched.")

if __name__ == '__main__':
    main()
