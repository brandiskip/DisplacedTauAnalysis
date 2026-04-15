import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import coffea.nanoevents.methods.vector as vector
from coffea.dataset_tools import preprocess

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

def save_comparison_overlay(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix="", log_y=False, normalize=True):
    import numpy as np
    
    if np.sum(h.values()) == 0:
        print(f"Skipping {var_name} (Empty)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for label in h.axes["cat"]:
        h_slice = h[{"cat": label}]
        total_events = np.sum(h_slice.values())
        
        if total_events == 0:
            continue
            
        if normalize:
            # Scale the histogram so the area under the curve is 1.0
            h_scaled = h_slice * (1.0 / total_events)
            h_scaled.plot1d(ax=ax, label=label)
        else:
            # Plot raw event counts
            h_slice.plot1d(ax=ax, label=label)
    
    ax.legend(title="Category")
    
    if normalize:
        ax.set_ylabel("Fraction of Events")
    else:
        ax.set_ylabel("Events")
        
    ax.set_title(f"Comparison: {var_name} {title_suffix}")
    
    if log_y:
        ax.set_yscale("log")
    
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{filename_suffix}.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved comparison plot to: {outpath}")

def save_2d_plot(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix=""):
    import numpy as np
    if np.sum(h.values()) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot the 2D colormap
    h.plot2d(ax=ax, cmap="viridis")
    
    ax.set_title(f"dR > 3.0: {var_name} {title_suffix}")
    
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{filename_suffix}_2D.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved 2D plot to: {outpath}")

def delta_r_mb2_prop(reco_obj, gen_obj):
    """
    Calculates dR using the propagated fields
    """
    dphi = np.abs(reco_obj.phi_at_mb2 - gen_obj.prop_phi_at_mb2)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = reco_obj.eta_at_mb2 - gen_obj.prop_eta_at_mb2
    return np.sqrt(deta**2 + dphi**2)

def delta_r_mb2_std(reco_obj, gen_obj):
    """
    Calculates dR using the standard unpropagated fields
    """
    dphi = np.abs(reco_obj.phi_at_mb2 - gen_obj.phi_at_mb2)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = reco_obj.eta_at_mb2 - gen_obj.eta_at_mb2
    return np.sqrt(deta**2 + dphi**2)

class SingleMuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            "n_events_initial": 0,
            "n_duplicates_removed": 0,
            
            "single_muon_pt": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, 0, 100, name="val", label=r"Single Muon $p_T$ [GeV]")),
            "single_muon_eta": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -2.5, 2.5, name="val", label=r"Single Muon $\eta$")),
            "single_muon_phi": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"Single Muon $\phi$")),
            "single_muon_dxy": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -100, 100, name="val", label=r"Single Muon $d_{xy}$")),
            "single_muon_dz_overlay": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, -300, 300, name="val", label=r"Single Muon $d_z$ [cm]")),
            "single_muon_validDTHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid DT Hits")),
            "single_muon_validCSCHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid CSC Hits")),
            "single_muon_validHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(80, 0, 80, name="val", label="Total Valid Muon Hits")),
            "single_muon_dtStations": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(6, 0, 6, name="val", label="DT Stations with Valid Hits")),
            "single_muon_dR_mb2": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, 0, 6, name="val", label=r"$\Delta R$ at MB2 (Reco vs Gen)")),
            "single_muon_timeAtIpInOut": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(25, -60, 60, name="val", label="Time at IP InOut [ns]")),
            "single_muon_timeAtIpInOutErr": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 3, name="val", label="Time at IP InOut Error [ns]")),
            "debug_dr3_eta_mb2": Hist(axis.StrCategory([], name="cat", label="Source", growth=True), axis.Regular(110, -105, 5, name="val", label=r"$\eta$ at MB2 ($\Delta R > 3.0$)")),
            "debug_dr3_phi_mb2": Hist(axis.StrCategory([], name="cat", label="Source", growth=True), axis.Regular(110, -105, 5, name="val", label=r"$\phi$ at MB2 ($\Delta R > 3.0$)")),
            "debug_dr3_eta_2d": Hist(axis.Regular(50, -3, 3, name="gen_eta", label=r"Gen $\eta$ at MB2"), axis.Regular(50, -3, 3, name="reco_eta", label=r"Reco $\eta$ at MB2")),
            "debug_dr3_phi_2d": Hist(axis.Regular(50, -3.2, 3.2, name="gen_phi", label=r"Gen $\phi$ at MB2"), axis.Regular(50, -3.2, 3.2, name="reco_phi", label=r"Reco $\phi$ at MB2")),

            "two_muons_cos_alpha": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(200, -1.01, 1.01, name="val", label=r"$\cos\alpha$ (Upper vs Lower)")),
        }

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")
        self.output["n_events_initial"] += len(events)
        
        events["DisMuon"] = ak.zip(
            {
                "pt": events.DisMuon.pt,
                "eta": events.DisMuon.eta,
                "phi": events.DisMuon.phi,
                "mass": events.DisMuon.mass,
                "charge": events.DisMuon.charge,          
                "timeNDof": events.DisMuon.timeNDof,
                "isStandalone": events.DisMuon.isStandalone,
                "dxy": events.DisMuon.dxy,
                "dz": events.DisMuon.dz,
                "numberOfValidMuonDTHits": events.DisMuon.numberOfValidMuonDTHits,
                "numberOfValidMuonCSCHits": events.DisMuon.numberOfValidMuonCSCHits,
                "numberOfValidMuonHits": events.DisMuon.numberOfValidMuonHits,
                "dtStationsWithValidHits": events.DisMuon.dtStationsWithValidHits,
                "timeAtIpInOut": events.DisMuon.timeAtIpInOut,
                "timeAtIpInOutErr": events.DisMuon.timeAtIpInOutErr,
                "mediumId": events.DisMuon.mediumId,
                "pfRelIso03_all": events.DisMuon.pfRelIso03_all,
                "eta_at_mb2": events.DisMuon.eta_at_mb2,
                "phi_at_mb2": events.DisMuon.phi_at_mb2,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        genpart_dict = {
            "pdgId": events.GenPart.pdgId,
            "status": events.GenPart.status,
            "eta_at_mb2": events.GenPart.eta_at_mb2, 
            "phi_at_mb2": events.GenPart.phi_at_mb2,
        }

        # Add propagated fields ONLY if they exist in the file
        if "prop_eta_at_mb2" in events.GenPart.fields:
            genpart_dict["prop_eta_at_mb2"] = events.GenPart.prop_eta_at_mb2
            genpart_dict["prop_phi_at_mb2"] = events.GenPart.prop_phi_at_mb2
        elif "GenPart_prop_eta_at_mb2" in events.fields:
            genpart_dict["prop_eta_at_mb2"] = events["GenPart_prop_eta_at_mb2"]
            genpart_dict["prop_phi_at_mb2"] = events["GenPart_prop_phi_at_mb2"]

        events["GenPart"] = ak.zip(
            genpart_dict,
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        gen_muons = events.GenPart[
            (abs(events.GenPart.pdgId) == 13) & 
            (events.GenPart.status == 1) #& (abs(events.GenPart.distinctParent.distinctParent.pdgId) == 1000015)
        ]
        
        dis_muons = events.DisMuon
        n_dismuons = ak.num(dis_muons)

        # ==========================================
        # 1. SINGLE MUON LOGIC
        # ==========================================
        # Require exactly 1 DisMuon
        mask_single = (n_dismuons == 1) & (ak.num(gen_muons) >= 1)
        
        if ak.sum(mask_single) > 0:
            single_muons = events.DisMuon[mask_single][:, 0]
            valid_gen_muons = gen_muons[mask_single]
            
            mask_medium = (single_muons.mediumId == True)
            mask_iso = (single_muons.pfRelIso03_all < 0.18)
            
            single_muons = single_muons[mask_medium & mask_iso]
            valid_gen_muons = valid_gen_muons[mask_medium & mask_iso]

            # Cosmic Logic
            if dataset.startswith("LooseMu") or dataset == "test_cosmics_calib":
                mask_upper_single = single_muons.phi > 0
                mask_lower_single = single_muons.phi < 0
                upper_singles = single_muons[mask_upper_single]
                lower_singles = single_muons[mask_lower_single]
                if len(upper_singles) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat="Upper Cosmic", val=upper_singles.dz)
                    self.output["single_muon_pt"].fill(cat="Upper Cosmic", val=upper_singles.pt)
                    self.output["single_muon_eta"].fill(cat="Upper Cosmic", val=upper_singles.eta)
                    self.output["single_muon_phi"].fill(cat="Upper Cosmic", val=upper_singles.phi)
                    self.output["single_muon_dxy"].fill(cat="Upper Cosmic", val=upper_singles.dxy)
                    self.output["single_muon_validDTHits"].fill(cat="Upper Cosmic", val=upper_singles.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat="Upper Cosmic", val=upper_singles.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat="Upper Cosmic", val=upper_singles.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat="Upper Cosmic", val=upper_singles.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat="Upper Cosmic", val=upper_singles.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat="Upper Cosmic", val=upper_singles.timeAtIpInOutErr)
                if len(lower_singles) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat="Lower Cosmic", val=lower_singles.dz)
                    self.output["single_muon_pt"].fill(cat="Lower Cosmic", val=lower_singles.pt)
                    self.output["single_muon_eta"].fill(cat="Lower Cosmic", val=lower_singles.eta)
                    self.output["single_muon_phi"].fill(cat="Lower Cosmic", val=lower_singles.phi)
                    self.output["single_muon_dxy"].fill(cat="Lower Cosmic", val=lower_singles.dxy)
                    self.output["single_muon_validDTHits"].fill(cat="Lower Cosmic", val=lower_singles.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat="Lower Cosmic", val=lower_singles.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat="Lower Cosmic", val=lower_singles.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat="Lower Cosmic", val=lower_singles.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat="Lower Cosmic", val=lower_singles.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat="Lower Cosmic", val=lower_singles.timeAtIpInOutErr)
            
            # Signal Logic
            elif dataset == "Stau_300_100mm":
                
                # Propagated Logic
                dr_mb2_prop_array = delta_r_mb2_prop(single_muons, valid_gen_muons)
                min_dr_mb2_prop = ak.min(dr_mb2_prop_array, axis=1)
                mask_dr_prop = ak.fill_none(min_dr_mb2_prop < 0.4, False)
                matched_muons_prop = single_muons[mask_dr_prop]
                
                # Fill Propagated Matches
                if len(matched_muons_prop) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat="Signal (Propagated)", val=matched_muons_prop.dz)
                    self.output["single_muon_pt"].fill(cat="Signal (Propagated)", val=matched_muons_prop.pt)
                    self.output["single_muon_eta"].fill(cat="Signal (Propagated)", val=matched_muons_prop.eta)
                    self.output["single_muon_phi"].fill(cat="Signal (Propagated)", val=matched_muons_prop.phi)
                    self.output["single_muon_dxy"].fill(cat="Signal (Propagated)", val=matched_muons_prop.dxy)
                    self.output["single_muon_validDTHits"].fill(cat="Signal (Propagated)", val=matched_muons_prop.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat="Signal (Propagated)", val=matched_muons_prop.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat="Signal (Propagated)", val=matched_muons_prop.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat="Signal (Propagated)", val=matched_muons_prop.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat="Signal (Propagated)", val=matched_muons_prop.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat="Signal (Propagated)", val=matched_muons_prop.timeAtIpInOutErr)
                
        # Filter for at least 2 DisMuons AND at least 1 GenMuon (Event-level filter)
        mask_multiple_disMuon_event = (ak.num(events.DisMuon) >= 2) & (ak.num(gen_muons) >= 1)
        
        if ak.sum(mask_multiple_disMuon_event) > 0:
            events = events[mask_multiple_disMuon_event]
            gen_muons = gen_muons[mask_multiple_disMuon_event]

            '''
            # This is for removing "duplicate" tracks
            #######################################################################################################
            sorted_pt = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
            lead = sorted_pt[:, 0]
            sublead = sorted_pt[:, 1]
            
            deta = lead.eta - sublead.eta
            dphi = lead.delta_phi(sublead)
            dpt = lead.pt - sublead.pt
            
            mask_same = (lead.charge * sublead.charge) > 0
            is_duplicate = mask_same & (abs(deta) < 0.01) & (abs(dphi) < 0.001) & (abs(dpt) < 0.5)
            
            self.output["n_duplicates_removed"] += ak.sum(is_duplicate)

            # Filter all arrays used later
            events = events[~is_duplicate]
            gen_muons = gen_muons[~is_duplicate]
            lead = lead[~is_duplicate]
            sublead = sublead[~is_duplicate]
            dis_muons = events.DisMuon
            #######################################################################################################
            '''
            dis_muons = events.DisMuon

            # Split into candidates based on Phi hemisphere
            # Upper: phi > 0, Lower: phi < 0
            upper_candidates = dis_muons[dis_muons.phi > 0]
            lower_candidates = dis_muons[dis_muons.phi < 0]

            # Count how many valid candidates exist in each hemisphere
            n_upper = ak.num(upper_candidates)
            n_lower = ak.num(lower_candidates)
            n_total = ak.num(dis_muons)

            # Criteria: At least 2 muons, AND (All Upper OR All Lower)
            mask_same_side = (n_total >= 2) & ((n_upper == n_total) | (n_lower == n_total))
            
            # Only process if such events exist to avoid errors
            if ak.sum(mask_same_side) > 0:
                # Select the muons from these specific events
                same_side_muons = dis_muons[mask_same_side]
                
                # Sort them by pT descending so we can distinguish Leading vs Subleading
                same_side_sorted = same_side_muons[ak.argsort(same_side_muons.pt, axis=1, ascending=False)]

                dr_same_side = same_side_sorted[:, 0].delta_r(same_side_sorted[:, 1])   
                mask_has_third = ak.num(same_side_sorted) >= 3
                
            # has at least 3 muons
            has_3_muons = (n_total >= 3)
            if ak.any(has_3_muons):
                print(f"WARNING: Events with >=3 muons")

            has_both_legs = (n_upper >= 1) & (n_lower >= 1)
            
            # Apply mask to 'events' and auxiliary arrays to keep them synchronized
            events = events[has_both_legs]
            gen_muons = gen_muons[has_both_legs]
            dis_muons = dis_muons[has_both_legs]

            #lead = lead[has_both_legs]          
            #sublead = sublead[has_both_legs]
            
            # Apply mask to our candidate arrays so we can pick the best ones from the valid events
            upper_candidates = upper_candidates[has_both_legs]
            lower_candidates = lower_candidates[has_both_legs]

            upper_candidates = upper_candidates[ak.argsort(upper_candidates.pt, axis=1, ascending=False)]
            lower_candidates = lower_candidates[ak.argsort(lower_candidates.pt, axis=1, ascending=False)]

            # Now we have exactly one Upper and one Lower muon per valid event
            upper = upper_candidates[:, 0]
            lower = lower_candidates[:, 0]

            '''
            ndof_quality = (upper.timeNDof > 7) & (lower.timeNDof > 7)
            lower_eta = (abs(lower.eta) < 0.7)
            lower_phi = (lower.phi > -3 * np.pi / 4) & (lower.phi < -np.pi / 4)
            lower_pt = (lower.pt > 12.5)
            upper_pt  = (upper.pt > 3.5)
            lower_dt_hits = (lower.numberOfValidMuonDTHits > 30)
            upper_dz = (abs(upper.dz) < 200)

            #lower_standalone = lower.isStandalone
            #upper_standalone = upper.isStandalone

            final_quality_mask = ndof_quality & lower_eta & lower_phi & lower_pt & upper_pt & lower_dt_hits & upper_dz #& lower_standalone & upper_standalone

            events = events[final_quality_mask]
            gen_muons = gen_muons[final_quality_mask]
            dis_muons = dis_muons[final_quality_mask]

            lead = lead[final_quality_mask]          
            sublead = sublead[final_quality_mask]

            upper = upper[final_quality_mask]
            lower = lower[final_quality_mask]
            '''

            # Calculate cosA based on upper and lower muons instead of leading pT muons
            dot_product = upper.px * lower.px + upper.py * lower.py + upper.pz * lower.pz
            denominator = upper.p * lower.p
            cosA = ak.where(denominator != 0, dot_product / denominator, -1000.0)

            #exact_two_mask = ak.num(dis_muons) == 2
            
            if dataset.startswith("LooseMu") or dataset == "test_cosmics_calib":
                self.output["two_muons_cos_alpha"].fill(
                        cat=dataset,
                        val=cosA
                    )
                '''
                if ak.any(exact_two_mask):
                    self.output["two_muons_cos_alpha"].fill(
                        cat=dataset,
                        val=cosA[exact_two_mask]
                    )
                '''    

        return self.output

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    
    '''
    fileset = {
        "test_cosmics_calib": [
            "root://eoscms.cern.ch//eos/cms/store/user/fiorendi/displacedTaus/test_cosmics_calib/nano_fix_calibrations.root"
        ]
    }

    print("Starting Processor for test_cosmics_calib...")
    executor = processor.FuturesExecutor(workers=8)
    runner = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
    )
    '''
    
    # Load the preprocessed Cosmic pickle file
    cosmic_pkl = "samples/Summer22_CHS_v17_Cosmic/Cosmic_preprocessed.pkl"
    print(f"Loading preprocessed Cosmics from {cosmic_pkl}...")
    with open(cosmic_pkl, "rb") as f:
        combined_runnable = pickle.load(f)
    
    # Load the preprocessed Signal pickle file
    signal_pkl = "samples/Signal/Stau_300_100mm_preprocessed.pkl"
    print(f"Loading preprocessed Signal from {signal_pkl}...")
    with open(signal_pkl, "rb") as f:
        signal_runnable = pickle.load(f)

    # Merge the dictionaries
    combined_runnable.update(signal_runnable)
    
    # Run the Processor
    #print("Starting Processor with combined Signal and Cosmic samples...")
    print("Starting Processor for test_cosmics_calib...")
    executor = processor.FuturesExecutor(workers=8)
    runner = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
    )
    '''
    out = runner(
        fileset,
        treename="Events",
        processor_instance=SingleMuonProcessor(),
    )
    '''
    out = runner(
        combined_runnable,
        treename="Events",
        processor_instance=SingleMuonProcessor(),
    )
    
    OUTPUT_DIR = "single_muon_signal_vs_cosmic_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Apply formatting to the titles and filenames
    PREFIX = "single_muon_"
    TITLE_MODIFIER = "(300 GeV 100 mm vs Cosmics)"
    FILE_MODIFIER = "Stau_300_100mm_overlay"

    overlay_plots = [
        "single_muon_dz_overlay",
        "single_muon_validDTHits",
        "single_muon_validCSCHits",
        "single_muon_validHits",
        "single_muon_dtStations",
        "single_muon_pt",
        "single_muon_eta",
        "single_muon_phi",
        "single_muon_dxy",
        "single_muon_dR_mb2",
        "single_muon_timeAtIpInOut",
        "single_muon_timeAtIpInOutErr",
        "two_muons_cos_alpha"
    ]

    for key, hist_obj in out.items():
        if isinstance(hist_obj, (int, float)): 
            continue
  
        if key in overlay_plots:
            # Set normalize=False so you can clearly see the raw stats shift
            save_comparison_overlay(hist_obj, key, PREFIX, OUTPUT_DIR, title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER, normalize=True)
    
    print("Done!")