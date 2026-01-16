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
        # Define Axis
        self.dxy_axis = axis.Variable(np.arange(0, 60, 5), name="dxy", label=r"$d_{xy}$ [cm]")
    
        # H1: Standard Jet Match (Denom = All)
        self.h_den_all = Hist(self.dxy_axis)
        self.h_num_jet_match_all = Hist(self.dxy_axis)
        
        # H2: Jet Match w/ Pion (Denom = Taus w/ Pion)
        self.h_den_pion = Hist(self.dxy_axis)
        self.h_num_jet_match_pion = Hist(self.dxy_axis)
        
        # H3: Pion Match (Denom = Taus w/ Pion)
        self.h_num_pion_match = Hist(self.dxy_axis)

    @property
    def accumulator(self):
        return {
            "h_den_all": self.h_den_all,
            "h_num_jet_match_all": self.h_num_jet_match_all,
            "h_den_pion": self.h_den_pion,
            "h_num_jet_match_pion": self.h_num_jet_match_pion,
            "h_num_pion_match": self.h_num_pion_match,
        }

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # Gen Selection (Denominator ALL Taus)
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

        # Mask: Exactly 1 valid GenTau and 1 GenMuon
        mask_all = (ak.num(gen_vis_taus) == 1) & (ak.num(gen_muons) == 1)
        events_all = events[mask_all]

        # Calculate dxy for ALL passing events
        tau_all = ak.firsts(events_all.GenVisTau[
             (abs(events_all.GenVisTau.parent.pdgId) == 15) &
             (events_all.GenVisTau.pt > min_pT)
        ])
        
        tau_parent = tau_all.parent
        dxy_all = abs(
            (tau_parent.vy - events_all.GenVtx.y) * np.cos(tau_parent.phi) - 
            (tau_parent.vx - events_all.GenVtx.x) * np.sin(tau_parent.phi)
        )

        # FILL DENOMINATOR 1 (All Taus)
        self.h_den_all.fill(dxy_all)

        pions_from_taus = get_gen_pions_from_taus(events_all)
        has_pion = ak.num(pions_from_taus, axis=1) > 0
        
        events_pion = events_all[has_pion]
        dxy_pion = dxy_all[has_pion]

        # FILL DENOMINATOR 2 & 3 (Taus w/ Pions)
        self.h_den_pion.fill(dxy_pion)

        # Process "All Events" for Plot 1 (Standard Jet Match)
        jets_all = select_jets(events_all)
        leading_jet_all = get_leading_jet(jets_all)
        has_leading_jet_all = ak.num(leading_jet_all) > 0
        
        tau_obj_all = tau_all 
        
        match_mask_all = np.zeros(len(events_all), dtype=bool)
        
        if len(events_all) > 0:
        
            jets_jagged_sub = leading_jet_all[has_leading_jet_all]
            taus_sub = tau_obj_all[has_leading_jet_all]

            jets_existing = ak.flatten(jets_jagged_sub, axis=1)
            
            dr_std = jets_existing.delta_r(taus_sub)
            matched_indices = (dr_std < 0.4)
            
            temp_mask = np.zeros(np.sum(has_leading_jet_all), dtype=bool)
            temp_mask[matched_indices] = True
            match_mask_all[has_leading_jet_all] = temp_mask

        # Fill numerator Jet matched / All
        self.h_num_jet_match_all.fill(dxy_all[match_mask_all])

        # Process "Pion Events" for Plot 2 & 3
        if len(events_pion) > 0:
            jets_pion = select_jets(events_pion)
            leading_jet_pion = get_leading_jet(jets_pion)
            reco_pf_pion = get_charged_pf(leading_jet_pion, min_reco_pf_pt)
            
            has_leading_jet_pion = ak.num(leading_jet_pion) > 0
            has_pf_in_jet = ak.any(ak.num(reco_pf_pion, axis=2) > 0, axis=1)
            
            # Recalculate Tau Obj 
            tau_obj_pion = ak.firsts(events_pion.GenVisTau[
                 (abs(events_pion.GenVisTau.parent.pdgId) == 15) &
                 (events_pion.GenVisTau.pt > min_pT)
            ])
            
            match_mask_pion_jet = np.zeros(len(events_pion), dtype=bool)
            
            if np.any(has_leading_jet_pion):
                 jets_jagged_sub = leading_jet_pion[has_leading_jet_pion]
                 taus_sub = tau_obj_pion[has_leading_jet_pion]
                 
                 jets_flat = ak.flatten(jets_jagged_sub, axis=1)
                 
                 dr_std_pion = jets_flat.delta_r(taus_sub)
                 temp_mask_pion = (dr_std_pion < 0.4)
                 match_mask_pion_jet[has_leading_jet_pion] = temp_mask_pion
            
            # Fill numerator Jet matched / Taus w/ Pion
            self.h_num_jet_match_pion.fill(dxy_pion[match_mask_pion_jet])

            # Process Specific Pion Match
            pf_candidates_mask = has_leading_jet_pion & has_pf_in_jet
            
            match_mask_pion_specific = np.zeros(len(events_pion), dtype=bool)
            
            if np.any(pf_candidates_mask):
                events_reco = events_pion[pf_candidates_mask]
                
                pions_final = get_gen_pions_from_taus(events_reco)
                leading_gen_pion = pions_final[ak.argsort(pions_final.pt, ascending=False)][:, 0:1]
                
                jets_reco = select_jets(events_reco)
                lead_jet_reco = get_leading_jet(jets_reco)
                reco_pf_final = get_charged_pf(lead_jet_reco, min_reco_pf_pt)
                leading_reco_pf = reco_pf_final[ak.argsort(reco_pf_final.pt, axis=2, ascending=False)][:, :, 0:1]
                
                flat_leading_pf = ak.flatten(leading_reco_pf, axis=1)
                
                dr_std_pion_match = flat_leading_pf.delta_r(leading_gen_pion)
                is_pion_match = ak.any(dr_std_pion_match < 0.4, axis=-1)
                
                match_mask_pion_specific[pf_candidates_mask] = is_pion_match

            # Fill numerator for Pion Matched / Taus w/ Pion
            self.h_num_pion_match.fill(dxy_pion[match_mask_pion_specific])

        #return self.accumulator
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
            # Extract values
            num_vals = h_num.values()
            den_vals = h_den.values()
            
            # Calculate Ratio and Errors using hist.intervals
            # efficiency_type="efficiency" gives Clopper-Pearson intervals
            ratio = np.divide(num_vals, den_vals, out=np.zeros_like(num_vals), where=den_vals!=0)
            yerr = intervals.ratio_uncertainty(num_vals, den_vals, uncertainty_type='efficiency')
            
            # Get Centers
            centers = h_num.axes[0].centers
            
            ax.errorbar(centers, ratio, yerr=yerr, fmt='o', color=color, label=label, capsize=3)
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            ax.set_ylabel("Efficiency")
            ax.set_xlabel(h_num.axes[0].label)

        # Generate Plots
        print("Generating Efficiency Plots...")

        # Plot 1: Standard Jet Match
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_ratio_internal(results['h_num_jet_match_all'], results['h_den_all'], ax, label="Jet Match (All Taus)")
        ax.set_title("Efficiency: Jet Match (All GenVisTaus)")
        ax.legend()
        plt.savefig("eff_jet_match_all.pdf")
        plt.close()
        print("Saved eff_jet_match_all.pdf")

        # Plot 2: Jet Match (Given Pion)
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_ratio_internal(results['h_num_jet_match_pion'], results['h_den_pion'], ax, label="Jet Match (Taus w/ Pion)")
        ax.set_title("Efficiency: Jet Match (Taus w/ GenPion)")
        ax.legend()
        plt.savefig("eff_jet_match_pion.pdf")
        plt.close()
        print("Saved eff_jet_match_pion.pdf")

        # Plot 3: Pion Match (Given Pion)
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_ratio_internal(results['h_num_pion_match'], results['h_den_pion'], ax, label="Pion Match (Taus w/ Pion)")
        ax.set_title("Efficiency: Pion Match (RecPF -> GenPion)")
        ax.legend()
        plt.savefig("eff_pion_match.pdf")
        plt.close()
        print("Saved eff_pion_match.pdf")

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