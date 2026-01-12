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
min_reco_pf_pt = 10

# ----------------------------------------------------------------------
# Helper Functions (Exact Copy from Post-Doc Code Logic)
# ----------------------------------------------------------------------
def get_gen_pions_from_taus(event_):
    pions_from_taus_ = event_.GenPart[abs(event_.GenPart.pdgId) == 211]
    pions_from_taus_ = pions_from_taus_[abs(pions_from_taus_.distinctParent.pdgId) == 15]
    pions_from_taus_ = pions_from_taus_[abs(pions_from_taus_.distinctParent.distinctParent.pdgId) == 1000015]
    return pions_from_taus_

def get_gen_muons_from_taus(event_):
    muons_from_tau_ = event_.GenPart[(abs(event_.GenPart.pdgId) == 13) & (event_.GenPart.hasFlags("isLastCopy"))]
    muons_from_tau_ = muons_from_tau_[
        (muons_from_tau_.pt > min_pT) &
        (abs(muons_from_tau_.eta) < max_eta) &
        (abs(muons_from_tau_.distinctParent.distinctParent.pdgId) == 1000015)
    ]
    return muons_from_tau_

def select_jets(event_):
    jets_ = event_.Jet[(abs(event_.Jet.eta) < max_eta) & (event_.Jet.pt > min_jet_pt)]
    jets_ = jets_[
       (jets_.neHEF < 0.99) &
       (jets_.neEmEF < 0.9) &
       (jets_.chMultiplicity + jets_.neMultiplicity > 1) &
       (jets_.chHEF > 0.01) &
       (jets_.chMultiplicity > 0) &
       (jets_.muEF < 0.8) &
       (jets_.chEmEF < 0.8)
    ]
    jets_ = jets_[(jets_.muEF < 0.5)]
    return jets_

def get_leading_jet(jets_):
    sorted_jets_ = jets_[ak.argsort(jets_.disTauTag_score1, ascending=False)]
    leading_jet_ = sorted_jets_[:, :1]
    return leading_jet_

def get_charged_pf(leading_jet_, min_reco_pf_pt_):
    reco_pf_ = leading_jet_.constituents.pf
    reco_pf_ = reco_pf_[(reco_pf_.charge != 0)]
    reco_pf_ = reco_pf_[(reco_pf_.pt > min_reco_pf_pt_)]
    return reco_pf_

def delta_r_ecal(reco_obj, gen_obj):
    dphi = np.abs(reco_obj.phi_at_ecal - gen_obj.phi_at_ecal_pion)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = reco_obj.eta_at_ecal - gen_obj.eta_at_ecal_pion
    return np.sqrt(deta**2 + dphi**2)

# ----------------------------------------------------------------------
# Processor Class
# ----------------------------------------------------------------------
class StauEfficiencyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # --- 1. Gen Selection (Denominator) ---
        gen_vis_taus = events.GenVisTau[
            (abs(events.GenVisTau.parent.pdgId) == 15) &
            (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
            (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) &
            (events.GenVisTau.parent.hasFlags("fromHardProcess")) &
            (events.GenVisTau.pt > min_pT) &
            (abs(events.GenVisTau.eta) < max_eta)
        ]
        
        # Lxy cut
        tau_vx = gen_vis_taus.parent.vx - gen_vis_taus.parent.parent.vx
        tau_vy = gen_vis_taus.parent.vy - gen_vis_taus.parent.parent.vy
        Lxy = np.sqrt(tau_vx**2 + tau_vy**2)
        gen_vis_taus = gen_vis_taus[Lxy < maxLxy]

        gen_muons = get_gen_muons_from_taus(events)
        gen_ele = events.GenPart[(abs(events.GenPart.pdgId) == 11) & (events.GenPart.hasFlags("isLastCopy"))]

        # Mask: 1 Tau
        # We calculate nGen
        initial_mask = (ak.num(gen_vis_taus) == 1) & (ak.num(gen_muons) == 1)
        events_filtered = events[initial_mask]
        nGen = len(events_filtered)
        
        # Init Pass counters
        nPass_Standard = 0
        nPass_ECAL = 0

        if nGen > 0:
            # --- 2. Iterative Filtering (Post-Doc Style) ---
            
            # Filter A: Must have Gen Pions
            pions_from_taus = get_gen_pions_from_taus(events_filtered)
            counts_pions = ak.num(pions_from_taus, axis=1)
            events_filtered = events_filtered[counts_pions > 0]
            
            if len(events_filtered) > 0:
                # Recalculate objects on filtered events
                pions_from_taus = get_gen_pions_from_taus(events_filtered)
                
                # Filter B: Must have Leading Jet
                jets = select_jets(events_filtered)
                leading_jet = get_leading_jet(jets) # [Events, 1]
                
                # Check 1: Leading Jet Exists?
                # Check 2: Leading Jet has Charged PF > min_pt?
                
                # We use Pad+Flatten to safely check "Is there a jet?"
                leading_jet_flat = ak.flatten(leading_jet, axis=1) 
                # This works because get_leading_jet returns at most 1 jet. 
                # If 0 jets, flatten removes the event from validity checking below, 
                # but we need to keep event alignment. 
                
                # Better approach: Use counts
                has_leading_jet = ak.num(leading_jet) > 0
                
                # Get PF candidates from the leading jet
                reco_pf = get_charged_pf(leading_jet, min_reco_pf_pt)
                counts_pf = ak.num(reco_pf, axis=2) # [Events, Jets] -> count per jet
                has_pf_in_jet = ak.any(counts_pf > 0, axis=1)
                
                # Events that pass RECO requirements
                reco_pass_mask = has_leading_jet & has_pf_in_jet
                events_reco = events_filtered[reco_pass_mask]
                
                if len(events_reco) > 0:
                    # --- 3. Matching on Selected Events ---
                    
                    # Re-fetch objects for the subset of passing events
                    jets_final = select_jets(events_reco)
                    leading_jet_final = get_leading_jet(jets_final)
                    reco_pf_final = get_charged_pf(leading_jet_final, min_reco_pf_pt)
                    
                    gen_vis_taus_final = events_reco.GenVisTau[
                        (abs(events_reco.GenVisTau.parent.pdgId) == 15) &
                        (abs(events_reco.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
                        (events_reco.GenVisTau.pt > min_pT) &
                        (abs(events_reco.GenVisTau.eta) < max_eta)
                    ] # Simplified fetch, we know they exist from step 1
                    
                    pions_final = get_gen_pions_from_taus(events_reco)
                    
                    # Prepare Leading Reco PF (Highest pT)
                    # reco_pf_final is [Events, Jets, PFs]
                    # We want the highest pT PF from the leading jet
                    sorted_reco_pf = reco_pf_final[ak.argsort(reco_pf_final.pt, axis=2, ascending=False)]
                    leading_reco_pf = sorted_reco_pf[:, :, 0:1] # Keep dims: [Events, 1(Jet), 1(PF)]
                    
                    # Prepare Leading Gen Pion
                    sorted_gen_pions = pions_final[ak.argsort(pions_final.pt, ascending=False)]
                    leading_gen_pion = sorted_gen_pions[:, 0:1] # [Events, 1(Pion)]

                    # --- Method 1: Standard Nearest (Leading Jet <-> Gen Tau) ---
                    # We use the leading jet (flattened to get object) and gen tau (flattened)
                    jet_obj = ak.flatten(leading_jet_final, axis=1)
                    tau_obj = ak.firsts(gen_vis_taus_final)
                    
                    dr_std = jet_obj.delta_r(tau_obj)
                    nPass_Standard = ak.sum(dr_std < 0.4)

                    # --- Method 2: ECAL Matching (Snippet) ---
                    # Flatten Jet dim to get [Events, PFs]
                    flat_leading_pf = ak.flatten(leading_reco_pf, axis=1)
                    
                    dr_matrix = delta_r_ecal(
                        ak.cartesian({"pf": flat_leading_pf, "pion": leading_gen_pion}, nested=True).pf,
                        ak.cartesian({"pf": flat_leading_pf, "pion": leading_gen_pion}, nested=True).pion
                    )
                    
                    dr_min = ak.min(dr_matrix, axis=2)
                    
                    # Count matches
                    pass_ecal = ak.any(dr_min < 0.4, axis=-1)
                    nPass_ECAL = ak.sum(pass_ecal)

        return {
            dataset: {
                "nGen": nGen,
                "nPass_Standard": nPass_Standard,
                "nPass_ECAL": nPass_ECAL,
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

    fileset_cleaned = {}
    for name, data in stau_fileset.items():
        fileset_cleaned[name] = list(data["files"].keys())

    print("Starting Parallel Analysis (Iterative Filtering)...")
    
    iterative_run = Runner(
        executor=FuturesExecutor(compression=None, workers=8),
        schema=PFNanoAODSchema,
        chunksize=10_000, 
    )

    output = iterative_run(
        fileset_cleaned,
        treename="Events",
        processor_instance=StauEfficiencyProcessor(),
    )

    results_std = []
    results_ecal = []

    for name, val_dict in output.items():
        nGen = val_dict['nGen']
        nPass_Std = val_dict['nPass_Standard']
        nPass_ECAL = val_dict['nPass_ECAL']
        
        try:
            parts = name.split('_')
            mass = int(parts[1])
            lifetime = int(parts[2].replace('mm', ''))
        except:
            print(f"Skipping plot for {name}")
            continue

        eff_std = nPass_Std / nGen if nGen > 0 else 0
        eff_ecal = nPass_ECAL / nGen if nGen > 0 else 0
        
        print(f"{name}: Gen={nGen}, Std={eff_std:.4f}, ECAL={eff_ecal:.4f}")
        
        results_std.append({"mass": mass, "lifetime": lifetime, "efficiency": eff_std})
        results_ecal.append({"mass": mass, "lifetime": lifetime, "efficiency": eff_ecal})

    # Save Data
    with open("efficiency_strict_standard.json", "w") as f:
        json.dump(results_std, f, indent=4)
    with open("efficiency_strict_ecal.json", "w") as f:
        json.dump(results_ecal, f, indent=4)

    # Plotting Function
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

    print("Generating Plots...")
    make_grid_plot("efficiency_strict_standard.json", "grid_plot_strict_standard.pdf", "Eff: Standard + Charged Pion")
    make_grid_plot("efficiency_strict_ecal.json", "grid_plot_strict_ecal.pdf", "Eff: ECAL + Charged Pion")
    print("Done!")