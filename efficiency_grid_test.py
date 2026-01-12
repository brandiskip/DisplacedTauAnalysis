import os
import json
import warnings
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Coffea Imports
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from coffea.processor import Runner, FuturesExecutor

# Configuration
PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"
warnings.filterwarnings("ignore") 

# Settings 
max_eta = 2.4
maxLxy = 100
min_pT = 20
min_jet_pt = 20  

# ----------------------------------------------------------------------
# Processor Class
# ----------------------------------------------------------------------
class StauEfficiencyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # Helper Functions
        def get_gen_muons(ev):
            muons = ev.GenPart[(abs(ev.GenPart.pdgId) == 13) & (ev.GenPart.hasFlags("isLastCopy"))]
            return muons[
                (muons.pt > min_pT) & (abs(muons.eta) < max_eta) &
                (abs(muons.distinctParent.distinctParent.pdgId) == 1000015)
            ]

        def get_gen_vis_taus(ev):
            taus = ev.GenVisTau[
                (abs(ev.GenVisTau.parent.pdgId) == 15) &
                (abs(ev.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
                (ev.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) &
                (ev.GenVisTau.parent.hasFlags("fromHardProcess")) &
                (ev.GenVisTau.pt > min_pT) &
                (abs(ev.GenVisTau.eta) < max_eta)
            ]
            vx = taus.parent.vx - taus.parent.parent.vx
            vy = taus.parent.vy - taus.parent.parent.vy
            Lxy = np.sqrt(vx**2 + vy**2)
            return taus[Lxy < maxLxy]

        def select_jets(ev):
            jets = ev.Jet[(abs(ev.Jet.eta) < max_eta) & (ev.Jet.pt > min_jet_pt)]
            jets = jets[
               (jets.neHEF < 0.99) & (jets.neEmEF < 0.9) &
               (jets.chMultiplicity + jets.neMultiplicity > 1) &
               (jets.chHEF > 0.01) & (jets.chMultiplicity > 0) &
               (jets.chEmEF < 0.8) &
               (jets.muEF < 0.5) 
            ]
            return jets

        # Event Selection 
        gen_vis_taus = get_gen_vis_taus(events)
        gen_muons = get_gen_muons(events)
        
        # Require exactly 1 Gen Vis Tau, 1 Gen Muon
        event_mask = (ak.num(gen_vis_taus) == 1) & (ak.num(gen_muons) == 1)
        
        events_filt = events[event_mask]
        gen_vis_taus = gen_vis_taus[event_mask]
        
        # Denominator: Total Gen Vis Taus in selected events
        nGen = len(events_filt)
        nMatched = 0

        if nGen > 0:
            jets = select_jets(events_filt)

            # Take the single Gen Tau per event
            single_gen_tau = ak.firsts(gen_vis_taus)
            
            # Find nearest Jet to this Tau (within dR 0.4)
            matched_jet = single_gen_tau.nearest(jets, threshold=0.4)
            
            # Count how many Taus successfully found a match
            nMatched = ak.sum(~ak.is_none(matched_jet))

        return {
            dataset: {
                "nGen": nGen,
                "nMatched": nMatched,
            }
        }

    def postprocess(self, accumulator):
        return accumulator

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":

    stau_fileset = {
        "Stau_300_1mm": {
            "files": {
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root": "Events",
            }
        },
        "Stau_300_100mm": {
            "files": {
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
            }
        },
        "Stau_300_1000mm": {
            "files": {
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root": "Events",
            }
        },
        "Stau_100_100mm": {
            "files": {
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
            }
        },
        "Stau_100_1000mm": {
            "files": {
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
            }
        },
        "Stau_500_1000mm": {
            "files": {
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root": "Events",
            }
        },
    }

    # Clean File Structure
    fileset_cleaned = {}
    for name, data in stau_fileset.items():
        fileset_cleaned[name] = list(data["files"].keys())

    # Setup Runner
    print("Starting Parallel Analysis (FuturesExecutor, workers=8)...")
    iterative_run = Runner(
        executor=FuturesExecutor(compression=None, workers=8),
        schema=PFNanoAODSchema,
        chunksize=10_000, 
    )

    # Run
    output = iterative_run(
        fileset_cleaned,
        treename="Events",
        processor_instance=StauEfficiencyProcessor(),
    )

    # Extract Results
    results = []

    for name, val_dict in output.items():
        nGen = val_dict['nGen']
        nMatched = val_dict['nMatched']
        
        try:
            parts = name.split('_')
            mass = int(parts[1])
            lifetime = int(parts[2].replace('mm', ''))
        except:
            print(f"Skipping plot for {name}")
            continue

        eff = nMatched / nGen if nGen > 0 else 0
        
        print(f"{name}: Gen={nGen}, Matched={nMatched}, Eff={eff:.4f}")
        
        results.append({"mass": mass, "lifetime": lifetime, "efficiency": eff})

    # Save Data
    with open("efficiency_simple_matching.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plotting
    def make_grid_plot(json_file, output_pdf, title_text):
        if not os.path.exists(json_file): return
        with open(json_file, "r") as f: data = json.load(f)
        if not data: return

        masses = sorted(set(entry["mass"] for entry in data))
        lifetimes = sorted(set(entry["lifetime"] for entry in data))
        mass_idx = {m: i for i, m in enumerate(masses)}
        lifetime_idx = {lt: i for i, lt in enumerate(lifetimes)}

        Z = np.zeros((len(lifetimes), len(masses)))
        for entry in data:
            Z[lifetime_idx[entry["lifetime"]], mass_idx[entry["mass"]]] = entry["efficiency"]

        Z = Z[::-1]
        lifetimes_reversed = lifetimes[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = cm.get_cmap("plasma")
        cmap.set_under('white')
        im = ax.imshow(Z, cmap=cmap, norm=mcolors.Normalize(vmin=0.0001, vmax=1))

        ax.set_xticks(range(len(masses)))
        ax.set_xticklabels([str(m) for m in masses])
        ax.set_yticks(range(len(lifetimes)))
        ax.set_yticklabels([f"{lt} mm" for lt in lifetimes_reversed])
        ax.set_title(title_text)
        ax.set_xlabel("Mass [GeV]")
        ax.set_ylabel(r"$c\tau$ [mm]")

        for i in range(len(lifetimes)):
            for j in range(len(masses)):
                val = Z[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.5 else "black")

        plt.colorbar(im, ax=ax).set_label("Efficiency")
        plt.savefig(output_pdf)
        plt.close()
        print(f"Saved {output_pdf}")

    print("Generating Plot...")
    make_grid_plot("efficiency_simple_matching.json", "grid_plot_simple_matching.pdf", "Efficiency: GenTau.nearest(Jet, dR<0.4)")
    print("Done!")