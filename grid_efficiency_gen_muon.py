import os
import json
import warnings
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import hist
from hist import Hist, axis, intervals
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
min_reco_pf_pt = 0

# ----------------------------------------------------------------------
# Helper Functions
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

def calc_and_add_dxy_gen_pion(pions_from_taus_, events_):
    counts_pions_ = ak.num(pions_from_taus_, axis=1)
    pv_y_ = np.asarray(events_.PVBS.y)
    pv_x_ = np.asarray(events_.PVBS.x)
    
    pv_y_expanded_pions_ = ak.unflatten(np.repeat(pv_y_, np.asarray(counts_pions_)), counts_pions_)
    pv_x_expanded_pions_ = ak.unflatten(np.repeat(pv_x_, np.asarray(counts_pions_)), counts_pions_)
    
    pions_from_taus_["dxy"] = (pions_from_taus_.vy - pv_y_expanded_pions_) * np.cos(pions_from_taus_.phi) - \
                              (pions_from_taus_.vx - pv_x_expanded_pions_) * np.sin(pions_from_taus_.phi)

def select_jets(event_):
    jets_ = event_.Jet[(abs(event_.Jet.eta) < max_eta) & (event_.Jet.pt > min_jet_pt)]
    jets_ = jets_[
       (jets_.neHEF < 0.99) &
       (jets_.neEmEF < 0.9) &
       (jets_.chMultiplicity + jets_.neMultiplicity > 1) &
       (jets_.chMultiplicity > 0) &
       (jets_.muEF < 0.5) &
       (jets_.chEmEF < 0.8)
    ]
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
        # Variable Binning
        dxy_bins_low = np.arange(0, 8, 1)
        dxy_bins_med = np.arange(8, 20, 4)
        dxy_bins_high = np.arange(20, 110, 10)
        dxy_bins_eff = np.concatenate([dxy_bins_low, dxy_bins_med, dxy_bins_high])
        
        self.dxy_axis = axis.Variable(dxy_bins_eff, name="dxy", label=r"$d_{xy}$ [cm]")
       
        self.h_den_all = Hist(self.dxy_axis)
        self.h_num_jet_match_all = Hist(self.dxy_axis)
        self.h_den_pion = Hist(self.dxy_axis)
        self.h_num_jet_match_pion = Hist(self.dxy_axis)
        self.h_num_pion_match = Hist(self.dxy_axis)
        self.h_num_pion_match_ecal = Hist(self.dxy_axis)

        self.min_reco_pf_pt = min_reco_pf_pt
        self.min_gen_tau_pt = min_pT
        self.decayM = 0

    @property
    def accumulator(self):
        return {
            "h_den_all": self.h_den_all,
            "h_num_jet_match_all": self.h_num_jet_match_all,
            "h_den_pion": self.h_den_pion,
            "h_num_jet_match_pion": self.h_num_jet_match_pion,
            "h_num_pion_match": self.h_num_pion_match,
            "h_num_pion_match_ecal": self.h_num_pion_match_ecal,
        }

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # Gen Vis Tau Selection (DM0)
        gen_vis_taus = events.GenVisTau[
            (abs(events.GenVisTau.parent.pdgId) == 15) &
            (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
            (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) &
            (events.GenVisTau.parent.hasFlags("fromHardProcess")) &
            (events.GenVisTau.status == self.decayM) &
            (events.GenVisTau.pt > self.min_gen_tau_pt) &
            (abs(events.GenVisTau.eta) < max_eta)
        ]

        # Lxy Cut
        tau_vx = gen_vis_taus.parent.distinctParent.vx - ak.firsts(gen_vis_taus.parent.distinctChildren.vx, axis =2)
        tau_vy = gen_vis_taus.parent.distinctParent.vy - ak.firsts(gen_vis_taus.parent.distinctChildren.vy, axis =2)
        Lxy = np.sqrt(tau_vx**2 + tau_vy**2)
        gen_vis_taus = ak.with_field(gen_vis_taus, Lxy, where="lxy")    
        gen_vis_taus = gen_vis_taus[abs(gen_vis_taus.lxy) < maxLxy]

        # Muon Selection & Event Mask
        muons_from_taus = get_gen_muons_from_taus(events)
        mask = (ak.num(gen_vis_taus) == 1) & (ak.num(muons_from_taus) == 1)
        
        filter_events = events[mask]
        filter_taus = gen_vis_taus[mask]
        
        # Gen Pion Selection
        pions_from_taus = get_gen_pions_from_taus(filter_events)
        counts_pions = ak.num(pions_from_taus, axis=1)
        pion_mask = (counts_pions > 0)
        
        # Apply pion mask to BOTH (Critical for alignment accuracy)
        filter_events = filter_events[pion_mask]
        filter_taus = filter_taus[pion_mask]
        
        if len(filter_events) == 0:
            return { dataset: self.accumulator }

        # RECO PRE-SELECTION (Must have Charged PF in Leading Jet)
        jets = select_jets(filter_events)
        leading_jet = get_leading_jet(jets)

        jets_one = ak.pad_none(jets, 1, axis=1)
        leading_one = jets_one[:, 0]
        empty_jet_lists = ak.Array([[[]]] * len(leading_one))
        
        charged_reco_pf = get_charged_pf(leading_jet, self.min_reco_pf_pt)
        pf_per_jet = ak.singletons(charged_reco_pf)

        reco_pf = ak.where(ak.is_none(leading_one), empty_jet_lists, pf_per_jet)

        pf_mask = (reco_pf.charge != 0) & (reco_pf.pt > self.min_reco_pf_pt)
        counts_charged_pf = ak.sum(pf_mask, axis=2)
        has_charged_in_leading = (ak.sum(counts_charged_pf, axis=1) > 0)
        has_charged_in_leading_event = ak.fill_none(ak.firsts(has_charged_in_leading, axis=1), False)

        # Require Reco Pion (and sync taus)
        pf_charged_events = filter_events[has_charged_in_leading_event]
        pf_charged_taus = filter_taus[has_charged_in_leading_event]
        
        if len(pf_charged_events) == 0:
            return { dataset: self.accumulator }

        # CALCULATE DXY (Denominator)
        pions_denom = get_gen_pions_from_taus(pf_charged_events)
        calc_and_add_dxy_gen_pion(pions_denom, pf_charged_events)
        
        sorted_pions_denom = pions_denom[ak.argsort(pions_denom.pt, ascending=False)]
        leading_pion_denom = ak.firsts(sorted_pions_denom)
        
        dxy_val = abs(leading_pion_denom.dxy)

        # FILL DENOMINATORS
        self.h_den_all.fill(dxy_val)
        self.h_den_pion.fill(dxy_val)

        # Gen Objects
        tau_obj = ak.firsts(pf_charged_taus) # From parallel filtering
        
        pions_final = get_gen_pions_from_taus(pf_charged_events)
        leading_gen_pion = pions_final[ak.argsort(pions_final.pt, ascending=False)][:, 0:1]

        # Reco Objects
        jets_final = select_jets(pf_charged_events)
        leading_jet_final = get_leading_jet(jets_final)
        
        # Object for Jet Matching
        leading_jet_obj = ak.firsts(leading_jet_final)

        # Object for Pion Matching
        reco_pf_final = get_charged_pf(leading_jet_final, self.min_reco_pf_pt)
        leading_reco_pf = reco_pf_final[ak.argsort(reco_pf_final.pt, axis=2, ascending=False)][:, :, 0:1]
        flat_leading_pf = ak.flatten(leading_reco_pf, axis=1)

        # ---------------------------------------------------------
        # NUMERATOR 1: JET MATCH (Standard dR < 0.4)
        # ---------------------------------------------------------
        dr_jet_tau = leading_jet_obj.delta_r(tau_obj)
        is_jet_match = ak.fill_none(dr_jet_tau < 0.4, False)
        
        self.h_num_jet_match_all.fill(dxy_val[is_jet_match])
        self.h_num_jet_match_pion.fill(dxy_val[is_jet_match])

        # ---------------------------------------------------------
        # NUMERATOR 2: PION MATCH (Standard dR < 0.4)
        # ---------------------------------------------------------
        dr_std_pion_match = flat_leading_pf.delta_r(leading_gen_pion)
        
        is_pion_match = ak.any(dr_std_pion_match < 0.4, axis=-1)
        is_pion_match = ak.fill_none(is_pion_match, False)

        self.h_num_pion_match.fill(dxy_val[is_pion_match])

        # ---------------------------------------------------------
        # NUMERATOR 3: PION MATCH (ECAL Match < 0.4)
        # ---------------------------------------------------------
        dr_ecal_val = delta_r_ecal(flat_leading_pf, leading_gen_pion)
        
        is_pion_match_ecal = ak.any(dr_ecal_val < 0.4, axis=-1)
        is_pion_match_ecal = ak.fill_none(is_pion_match_ecal, False)
        
        self.h_num_pion_match_ecal.fill(dxy_val[is_pion_match_ecal])

        return { dataset: self.accumulator }

    def postprocess(self, accumulator):
        return accumulator

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":

    stau_fileset = {
        # "Stau_300_1mm": {
        #     "files": {
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root": "Events",
        #     }
        # },
        
        "Stau_300_100mm": {
            "files": {
                "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
            }
        },

        # "Stau_300_1000mm": {
        #     "files": {
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root": "Events",
        #     }
        # },
        # "Stau_100_100mm": {
        #     "files": {
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
        #     }
        # },
        # "Stau_100_1000mm": {
        #     "files": {
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
        #     }
        # },
        # "Stau_500_1000mm": {
        #     "files": {
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root": "Events",
        #         "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root": "Events",
        #     }
        # },
    }

    fileset_cleaned = {}
    for name, data in stau_fileset.items():
        fileset_cleaned[name] = list(data["files"].keys())

    print("Starting Analysis...")
    iterative_run = Runner(
        executor=FuturesExecutor(compression=None, workers=4),
        schema=PFNanoAODSchema,
        chunksize=10_000, 
    )

    output = iterative_run(
        fileset_cleaned,
        treename="Events",
        processor_instance=StauEfficiencyProcessor(),
    )

    if "Stau_300_100mm" in output:
        results = output["Stau_300_100mm"]
        
        # Save results to pickle
        import pickle
        with open("dxy_efficiency_hists.pkl", "wb") as f:
            pickle.dump(results, f)
        
        print("\n--- Processing Complete ---")
        print("Results saved to dxy_efficiency_hists.pkl")

        # Plotting Function
        def plot_ratio_internal(h_num, h_den, ax, label=None, color='black'):
            num_vals = h_num.values()
            den_vals = h_den.values()
            
            ratio = np.divide(num_vals, den_vals, out=np.zeros_like(num_vals), where=den_vals!=0)
            yerr = intervals.ratio_uncertainty(num_vals, den_vals, uncertainty_type='efficiency')
            
            centers = h_num.axes[0].centers
            edges = h_num.axes[0].edges
            width = (edges[1:] - edges[:-1]) / 2

            ax.errorbar(centers, ratio, yerr=yerr, xerr=width, fmt='o', color=color, label=label, capsize=3)
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            ax.set_ylabel("Efficiency")
            ax.set_xlabel(h_num.axes[0].label)

        print("Generating Efficiency Plots...")

        # Standard Jet Match
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_ratio_internal(results['h_num_jet_match_all'], results['h_den_all'], ax, label="Jet Match (dR < 0.4)")
        ax.set_title("Eff: Jet Match GenVisTau (Standard) [Has Reco Pion]")
        ax.legend()
        plt.savefig("eff_jet_match_req_reco_dm0_no_pt_cut_pion.pdf")
        plt.close()
        print("Saved eff_jet_match_all.pdf")

        # Pion Match (STANDARD dR)
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_ratio_internal(results['h_num_pion_match'], results['h_den_pion'], ax, label="Pion Match (Standard dR < 0.4)")
        ax.set_title("Eff: RecoPion Match GenPion (Standard) [Has Reco Pion]")
        ax.legend()
        plt.savefig("eff_pion_match_req_reco_dm0_no_pt_cut_pion.pdf")
        plt.close()
        print("Saved eff_pion_match.pdf")

        # Pion Match (ECAL dR)
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_ratio_internal(results['h_num_pion_match_ecal'], results['h_den_pion'], ax, label="Pion Match (ECAL dR < 0.4)")
        ax.set_title("Eff: RecoPion Match GenPion (ECAL) [Has Reco Pion]")
        ax.legend()
        plt.savefig("eff_pion_match_ecal_req_reco_dm0_no_pt_cut_pion.pdf")
        plt.close()
        print("Saved eff_pion_match_ecal.pdf")

    '''
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
    make_grid_plot("efficiency_strict_standard.json", "grid_plot_strict_standard_no_pt_cut.pdf", "Eff: Standard + Charged Pion")
    make_grid_plot("efficiency_strict_ecal.json", "grid_plot_strict_ecal_no_pt_cut.pdf", "Eff: ECAL + Charged Pion")
    print("Done!")
    '''