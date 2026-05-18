import argparse
import importlib
import pickle
import time
import logging
from coffea.dataset_tools import preprocess
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None})
from uproot.exceptions import KeyInFileError
from dask.distributed import Client, LocalCluster
import sys      
import os        

sys.path.insert(0, os.path.abspath('.'))

# Dynamically generate all 16 signal sample names
SIGNAL_SAMPLES = [
    f"Stau_{mass}_{ctau}mm" 
    for mass in [100, 200, 300, 500] 
    for ctau in [1, 10, 100, 1000]
]

parser = argparse.ArgumentParser(description="Preprocess Signal samples")
parser.add_argument(
    "--sample",
    required=True, 
    choices=SIGNAL_SAMPLES + ["all"], 
    help='Specify the signal sample you want to process, or "all" to run sequentially'
)
parser.add_argument(
    "--nfiles",
    default='-1',
    required=False,
    help='Specify the number of input files to process (-1 for all)'
)
parser.add_argument(
    "--nanov",
    default='Signal_Samples', 
    required=False,
    help='Directory inside samples/ containing the filesets'
)
args = parser.parse_args()

outdir_p = f'{args.nanov}.'
outdir_s = f'{args.nanov}/'

# Global parameters
STEP_SIZE = 50_000
FILES_PER_BATCH = 10

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)    
    
    # Determine which samples to run
    samples_to_run = SIGNAL_SAMPLES if args.sample == "all" else [args.sample]
    nfiles = int(args.nfiles)

    print("Attempting to start Dask Cluster...")
    try:
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        client = Client(cluster)
        print(f"Cluster started! Dashboard link: {client.dashboard_link}")
    except Exception as e:
        print(f"\nCRITICAL ERROR starting cluster: {e}\n")
        exit(1)

    # Loop over the selected samples
    for current_sample in samples_to_run:
        tic = time.time()
        print(f"\n--- Starting preprocessing for {current_sample} ---")
        
        # Dynamically import the fileset module inside the loop
        module_path = f"samples.{outdir_p}fileset_{current_sample}"
        try:
            module = importlib.import_module(module_path)
            fileset = module.fileset
        except ModuleNotFoundError:
            print(f"Error: Could not find fileset at samples/{outdir_s}fileset_{current_sample}.py")
            continue # Skip to the next sample instead of crashing

        # Handle nfiles limit
        if nfiles != -1:
            for k in fileset.keys():
                if nfiles < len(fileset[k]['files']):
                    fileset[k]['files'] = dict(list(fileset[k]['files'].items())[:nfiles])

        print("Will process {} files from: {}".format(nfiles if nfiles != -1 else "ALL", list(fileset.keys())))

        try:
            dataset_runnable, dataset_updated = preprocess(
                fileset,
                align_clusters=False,
                step_size=STEP_SIZE,
                files_per_batch=FILES_PER_BATCH,
                skip_bad_files=True,
                save_form=False,
                file_exceptions=(OSError, KeyInFileError),
                allow_empty_datasets=False,
            )
        except Exception as e:
            print(f"\nCRITICAL ERROR during preprocessing {current_sample}: {e}\n")
            continue # Skip to the next sample

        pkl_name = f"samples/{outdir_s}{current_sample}_preprocessed.pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(dataset_runnable, f)

        elapsed = time.time() - tic 
        print(f"Saved preprocessed data to: {pkl_name}")
        print(f"Finished {current_sample} in {elapsed:.1f}s") 

    # Clean up cluster at the very end
    client.shutdown()
    cluster.close()
    print("\nAll requested samples finished!")