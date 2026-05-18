import subprocess
from pathlib import Path

SAMPLES = {
    "Stau_100_1mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_100_10mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_100_100mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_100_1000mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_200_1mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-200_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_200_10mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-200_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_200_100mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_200_1000mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-200_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_300_1mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_300_10mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_300_100mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_300_1000mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_500_1mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_500_10mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_500_100mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
    "Stau_500_1000mm": "/store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v19/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8",
}

# Output directory matches the --nanov default in preprocess.py
outdir = 'samples/Signal_Samples/'
XROOTD_PREFIX = "root://cmsxrootd.fnal.gov/"
EOS_LOC = 'root://cmseos.fnal.gov'

outfolder = Path(outdir)
outfolder.mkdir(parents=True, exist_ok=True)

current_path = Path()
for part in outfolder.parts:
    current_path = current_path / part
    (current_path / "__init__.py").touch(exist_ok=True)

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
    for sample_name, full_path in SAMPLES.items():
        outfile = f"{outdir}fileset_{sample_name}.py"
        
        print(f"\nScanning directory for {sample_name}: {full_path}")
        rootfiles = list_root_files(full_path)
        
        if not rootfiles:
            print(f"No root files found for {sample_name}! Check your proxy and path.")
            continue
            
        print(f"Found {len(rootfiles)} files. Writing fileset...")
        with open(outfile, "w") as f:
            f.write("fileset = {\n")
            f.write(f"    '{sample_name}': {{\n")
            f.write("        \"files\": {\n")
            for rf in rootfiles:
                f.write(f"            \"{XROOTD_PREFIX}{rf}\": \"Events\",\n")
            f.write("        }\n")
            f.write("    }\n")
            f.write("}\n")
        print(f"Success! Wrote fileset to {outfile}")

if __name__ == "__main__":
    main()