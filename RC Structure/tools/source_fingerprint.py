"""Print the source and environment fingerprint of this machine, or compare it with a saved baseline.

Every design record carries the SHA-256 of each source file a design depends on (Design_Driver.source_sha256:
Structure_Parameters.py, RC_Design_Check.py, Redesign.py and every .py under Design, Model, Loads and Analysis,
read as text so line endings do not matter). Machines that share a batch must agree on all of them, on the
generation chain's own files, and on the solver and library versions. This tool reports all three.

  python tools/source_fingerprint.py                      print this machine's fingerprint
  python tools/source_fingerprint.py --write baseline.json     save it
  python tools/source_fingerprint.py --check baseline.json     compare; exit code 1 on any difference
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The generation chain's own files: outside the design identity, but they decide what is run and exported.
CHAIN_FILES = ("Ground_Motion_Main.py", "Data_Generation/Run_Research_Batch.py", "Loads/Ground_Motion.py",
               "Data_Generation/Generate_Parameterized_Dataset.py",
               "Data_Generation/Generate_Hybrid_Dataset.py", "Data_Generation/Hybrid_Exporter.py",
               "Data_Generation/Graph_Exporter.py", "Data_Generation/Calibrate_Intensity.py")
PACKAGES = ("openseespy", "openseespywin", "numpy", "scipy", "matplotlib")


def text_sha256(path):
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def fingerprint():
    from Design import Design_Driver as driver
    import openseespy.opensees as ops
    design_source = driver.source_sha256()
    chain = {name: text_sha256(ROOT / name) for name in CHAIN_FILES if (ROOT / name).exists()}
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    aggregate = lambda mapping: hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest()   # noqa: E731
    return {"design_source_aggregate_sha256": aggregate(design_source), "design_source_files": len(design_source),
            "generation_chain_aggregate_sha256": aggregate(chain),
            "opensees_version": str(ops.version()), "python": platform.python_version(), "packages": packages,
            "platform": platform.platform(), "design_source": design_source, "generation_chain": chain}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    here = fingerprint()
    if args.write:
        args.write.write_text(json.dumps(here, indent=1), encoding="utf-8")
    if args.check:
        base = json.loads(args.check.read_text(encoding="utf-8"))
        problems = []
        for group in ("design_source", "generation_chain"):
            for name in sorted(set(base[group]) | set(here[group])):
                if base[group].get(name) != here[group].get(name):
                    problems.append(f"{group}: {name} differs" if name in base[group] and name in here[group]
                                    else f"{group}: {name} is {'missing here' if name in base[group] else 'extra here'}")
        for key in ("opensees_version", "python"):
            if base[key] != here[key]:
                problems.append(f"{key}: baseline {base[key]}, here {here[key]}")
        for name, version in base["packages"].items():
            if here["packages"].get(name) != version:
                problems.append(f"package {name}: baseline {version}, here {here['packages'].get(name)}")
        print("MATCHES the baseline" if not problems else "DIFFERS from the baseline:\n  - " + "\n  - ".join(problems))
        raise SystemExit(1 if problems else 0)
    print(json.dumps({key: value for key, value in here.items() if key not in ("design_source", "generation_chain")}, indent=1))


if __name__ == "__main__":
    main()
