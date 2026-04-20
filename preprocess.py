import argparse, importlib
import pickle
import time, logging
from coffea.dataset_tools import preprocess
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None})
from uproot.exceptions import KeyInFileError
from dask.distributed import Client, LocalCluster

parser = argparse.ArgumentParser(description="Preprocess Cosmic and Signal samples")
parser.add_argument(
    "--sample",
    default='Cosmic',
    choices=['Cosmic', 'Stau_100_100mm', 'Stau_300_100mm', 'Stau_500_100mm'], 
    help='Specify the sample you want to process')
parser.add_argument(
    "--nfiles",
    default='-1',
    required=False,
    help='Specify the number of input files to process (-1 for all)')
parser.add_argument(
    "--nanov",
    default='Summer22_CHS_v19_Cosmic', 
    required=False,
    help='Specify the custom nanoaod version to process')
args = parser.parse_args()

outdir_p = f'{args.nanov}.'
outdir_s = f'{args.nanov}/'

# ADDED 100 and 500 to the samples dictionary
samples = {
    "Cosmic" : f"samples.{outdir_p}fileset_Cosmic",
    "Stau_100_100mm" : f"samples.{outdir_p}fileset_Stau_100_100mm",
    "Stau_300_100mm" : f"samples.{outdir_p}fileset_Stau_300_100mm",
    "Stau_500_100mm" : f"samples.{outdir_p}fileset_Stau_500_100mm",
}

try:
    module = importlib.import_module(samples[args.sample])
    fileset = module.fileset
except ModuleNotFoundError:
    print(f"Error: Could not find the fileset at samples/{outdir_s}fileset_{args.sample}.py")
    print("Did you run the write_filesets script first?")
    exit(1)

nfiles = int(args.nfiles)
if nfiles != -1:
    for k in fileset.keys():
        if nfiles < len(fileset[k]['files']):
            fileset[k]['files'] = dict(list(fileset[k]['files'].items())[:nfiles])

print("Will process {} files from: {}".format(nfiles if nfiles != -1 else "ALL", fileset.keys()))

# ADDED 100 and 500 to the parameters dictionary
pars_per_sample = {
    "Cosmic" : [50_000, 10], 
    "Stau_100_100mm" : [50_000, 10],
    "Stau_300_100mm" : [50_000, 10], 
    "Stau_500_100mm" : [50_000, 10],
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)    
    tic = time.time()
    print("Attempting to start Dask Cluster...")
    try:
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        client = Client(cluster)
        print(f"Cluster started! Dashboard link: {client.dashboard_link}")
    except Exception as e:
        print(f"\nCRITICAL ERROR starting cluster: {e}\n")
        exit(1)

    print(f"Starting preprocessing for {args.sample}...")
    try:
        dataset_runnable, dataset_updated = preprocess(
           fileset,
           align_clusters=False,
           step_size=pars_per_sample[args.sample][0],
           files_per_batch=pars_per_sample[args.sample][1],
           skip_bad_files=True,
           save_form=False,
           file_exceptions=(OSError, KeyInFileError),
           allow_empty_datasets=False,
        )
    except Exception as e:
        print(f"\nCRITICAL ERROR during preprocessing: {e}\n")
        client.shutdown()
        cluster.close()
        exit(1)

    pkl_name = f"samples/{outdir_s}{args.sample}_preprocessed.pkl"
    with open(pkl_name, "wb") as f:
        pickle.dump(dataset_runnable, f)

    elapsed = time.time() - tic 
    print(f"Saved preprocessed data to: {pkl_name}")
    print(f"Finished in {elapsed:.1f}s") 
    client.shutdown()
    cluster.close()