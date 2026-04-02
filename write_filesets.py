import os
import subprocess
import json
from pathlib import Path

# The base directory for your signal sample
BASE_DIR = "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v18/"
SIGNAL_DIR = "SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8"

outdir = 'samples/Signal/'
XROOTD_PREFIX = "root://cmsxrootd.fnal.gov/"
EOS_LOC = 'root://cmseos.fnal.gov'

outfolder = Path(outdir)
if not outfolder.exists():
    outfolder.mkdir(parents=True, exist_ok=True)
    (outfolder / "__init__.py").touch(exist_ok=True)

outfile = f"{outdir}fileset_Stau_300_100mm.py"

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}\n{result.stderr}")
        return []
    return result.stdout.strip().split("\n")

def list_root_files(path):
    cmd = f"xrdfs {EOS_LOC} ls {path}"
    files = run_cmd(cmd)
    return [f for f in files if f.endswith(".root")]

def main():
    full_path = os.path.join(BASE_DIR, SIGNAL_DIR)
    print(f"Scanning directory: {full_path}")
    rootfiles = list_root_files(full_path)
    if not rootfiles:
        print("No root files found! Check your proxy and path.")
        return
    print(f"Found {len(rootfiles)} files. Writing fileset...")
    with open(outfile, "w") as f:
        f.write("fileset = {\n")
        f.write("    'Stau_300_100mm': {\n")
        f.write("        \"files\": {\n")
        for rf in rootfiles:
            f.write(f"            \"{XROOTD_PREFIX}{rf}\": \"Events\",\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("}\n")
    print(f"Success! Wrote fileset to {outfile}")

if __name__ == "__main__":
    main()