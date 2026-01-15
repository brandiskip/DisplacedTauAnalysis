import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hist
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
from coffea.analysis_tools import PackedSelection
import coffea.nanoevents.methods.vector as vector
import pickle

# --- Plotting Helper Function ---
def save_2d_plot(h, var_name, sample_name, PREFIX, OUTPUT_DIR):
    """Plots a 2D histogram with a logarithmic color scale."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    h.plot2d(
        ax=ax, 
        cmap="viridis", 
        norm=mcolors.LogNorm(vmin=1)
    )
    
    ax.set_title(f"{sample_name}: {var_name}")
    
    filename = f"{PREFIX}{var_name}_log.pdf"
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved Log-Z 2D plot to: {outpath}")

# --- The Processor Class ---
class CosmicProcessor(processor.ProcessorABC):
    def __init__(self):
        # Define histograms
        self.output = {
            "dt_vs_cosA_opp_charge": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t = t_{upper} - t_{lower}$ [ns]")
            ),
            "pt_lead_vs_sublead_opp_charge": Hist(
                axis.Regular(100, 0, 200, name="lead_pt", label="Leading Displaced Muon $p_T$ [GeV]"),
                axis.Regular(100, 0, 200, name="sublead_pt", label="Subleading Displaced Muon $p_T$ [GeV]"),
            ),
            # Placeholder for the commented out histogram logic (if you re-enable it)
            "dt_vs_cosA_veto": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t = t_{upper} - t_{lower}$ [ns]")
            ),
        }

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # 1. Attach vector behavior to DisMuon so we can use .px, .py, .pz
        # Note: In PFNanoAODSchema, we usually just map the collection
        events["DisMuon"] = ak.zip(
            {
                "pt": events.DisMuon.pt,
                "eta": events.DisMuon.eta,
                "phi": events.DisMuon.phi,
                "mass": events.DisMuon.mass,
                "charge": events.DisMuon.charge,
                "timeAtIpInOut": events.DisMuon.timeAtIpInOut,
                "timeNDof": events.DisMuon.timeNDof,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # ---------------------------------------------------------
        # COMMENTED OUT LOGIC (Replicated from your script)
        # ---------------------------------------------------------
        '''
        # Multiplicity filter (Need at least 2 muons)
        mask_2dis = ak.num(events.DisMuon) >= 2
        events_veto = events[mask_2dis]
        
        sorted_dis_muons = events_veto.DisMuon[ak.argsort(events_veto.DisMuon.pt, axis=1, ascending=False)]
        leading_dis_muon_reco = sorted_dis_muons[:, 0]
        subleading_dis_muon_reco = sorted_dis_muons[:, 1]

        sorted_DisMuons_phi = events_veto.DisMuon[ak.argsort(events_veto.DisMuon.phi, axis=1, ascending=False)]
        upper_muon = sorted_DisMuons_phi[:, 0]
        lower_muon = sorted_DisMuons_phi[:, 1]

        # dof mask
        ndof_quality = (upper_muon.timeNDof > 7) & (lower_muon.timeNDof > 7)
        
        # Calculate Delta_t and create mask
        delta_t_for_mask = (upper_muon.timeAtIpInOut - lower_muon.timeAtIpInOut)
        rejection_condition = (delta_t_for_mask < -20.0) & ndof_quality

        # Calculate cosAlpha
        dot_product_temp = leading_dis_muon_reco.px * subleading_dis_muon_reco.px + \
                           leading_dis_muon_reco.py * subleading_dis_muon_reco.py + \
                           leading_dis_muon_reco.pz * subleading_dis_muon_reco.pz
        den_temp = leading_dis_muon_reco.p * subleading_dis_muon_reco.p
        cosA_temp = ak.where(den_temp != 0, dot_product_temp / den_temp, -1000.0)

        # Final Mask logic
        mask_same_charge = (leading_dis_muon_reco.charge * subleading_dis_muon_reco.charge) > 0
        final_mask = mask_same_charge & (~rejection_condition) & (cosA_temp >= -0.99)
        events_veto = events_veto[final_mask]

        # RE-DERIVE variables from filtered events
        sorted_phi_final = events_veto.DisMuon[ak.argsort(events_veto.DisMuon.phi, axis=1, ascending=False)]
        u_muon_final = sorted_phi_final[:, 0]
        l_muon_final = sorted_phi_final[:, 1]

        delta_t = (u_muon_final.timeAtIpInOut - l_muon_final.timeAtIpInOut)

        # Re-calc cosA for plotting (using filtered events)
        sorted_pt_final = events_veto.DisMuon[ak.argsort(events_veto.DisMuon.pt, axis=1, ascending=False)]
        lead_final = sorted_pt_final[:, 0]
        subl_final = sorted_pt_final[:, 1]
        
        dot_prod_final = lead_final.px * subl_final.px + lead_final.py * subl_final.py + lead_final.pz * subl_final.pz
        den_final = lead_final.p * subl_final.p
        flat_cosA = ak.where(den_final != 0, dot_prod_final / den_final, -1000.0)

        self.output["dt_vs_cosA_veto"].fill(cosA=flat_cosA, dt=delta_t)
        '''

        # ---------------------------------------------------------
        # ACTIVE LOGIC
        # ---------------------------------------------------------
        
        # Filter for at least 2 muons
        mask_2dis = ak.num(events.DisMuon) >= 2
        events = events[mask_2dis]
        
        sorted_dis_muons = events.DisMuon[ak.argsort(events.DisMuon.pt, ascending=False)]
        leading_dis_muon_reco = sorted_dis_muons[:, 0]
        subleading_dis_muon_reco = sorted_dis_muons[:, 1]
        
        # mask_same_charge_dis = (leading_dis_muon_reco.charge * subleading_dis_muon_reco.charge) > 0
        # events = events[mask_same_charge_dis]

        mask_diff_charge_dis = (leading_dis_muon_reco.charge * subleading_dis_muon_reco.charge) < 0
        events = events[mask_diff_charge_dis]

        # Re-sort after filtering
        sorted_dis_muons = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        leading_dis_muon_reco = sorted_dis_muons[:, 0]
        subleading_dis_muon_reco = sorted_dis_muons[:, 1]

        # Sorted by phi for timing (Upper vs Lower)
        sorted_DisMuons_phi = events.DisMuon[ak.argsort(events.DisMuon.phi, axis=1, ascending=False)]
        upper_muon = sorted_DisMuons_phi[:, 0]
        lower_muon = sorted_DisMuons_phi[:, 1]

        time_upper = upper_muon.timeAtIpInOut
        time_lower = lower_muon.timeAtIpInOut
        delta_t = (time_upper - time_lower)

        # Calculate CosAlpha
        dot_product_dis = leading_dis_muon_reco.px * subleading_dis_muon_reco.px + \
                          leading_dis_muon_reco.py * subleading_dis_muon_reco.py + \
                          leading_dis_muon_reco.pz * subleading_dis_muon_reco.pz

        den_dis = leading_dis_muon_reco.p * subleading_dis_muon_reco.p

        cosA_dis = ak.where(den_dis != 0, dot_product_dis / den_dis, -1000.0)
        
        # Fill Histograms
        self.output["dt_vs_cosA_opp_charge"].fill(
            cosA=cosA_dis, 
            dt=delta_t
        )
        
        self.output["pt_lead_vs_sublead_opp_charge"].fill(
            lead_pt=leading_dis_muon_reco.pt, 
            sublead_pt=subleading_dis_muon_reco.pt
        )

        return self.output

    def postprocess(self, accumulator):
        return accumulator


# --- Main Execution Block ---
if __name__ == '__main__':
    
    # 1. Load the Preprocessed Data (The .pkl file you made)
    # Make sure this path matches exactly where your preprocess script saved it
    pkl_file_path = "samples/Summer22_CHS_v14_Cosmic/Cosmic_preprocessed.pkl"
    
    with open(pkl_file_path, "rb") as f:
        runnable = pickle.load(f)

    # 2. Run the Processor
    print("Starting Processor...")
    executor = processor.FuturesExecutor(workers=4)
    runner = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
    )

    out = runner(
        runnable,
        treename="Events",
        processor_instance=CosmicProcessor(),
    )

    # 3. Save Plots (Post-processing)
    OUTPUT_DIR = "cosmic_muon_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PREFIX_DISMUON = "cosmics_dismuon_"

    # Plot Active Histograms
    if "dt_vs_cosA_opp_charge" in out:
        save_2d_plot(
            out["dt_vs_cosA_opp_charge"], 
            "DisMuon_DeltaT_vs_CosAlpha_opp_charge_req", 
            "Cosmic", 
            PREFIX_DISMUON, 
            OUTPUT_DIR
        )

    if "pt_lead_vs_sublead_opp_charge" in out:
        save_2d_plot(
            out["pt_lead_vs_sublead_opp_charge"], 
            "DisMuon_Lead_vs_Sublead_pT_opp_charge_req", 
            "Cosmic", 
            PREFIX_DISMUON, 
            OUTPUT_DIR
        )

    # Plot Commented Histograms (if they were filled)
    # The 'if' check prevents errors if you leave the filling logic commented out
    if out["dt_vs_cosA_veto"].sum() > 0:
         save_2d_plot(
            out["dt_vs_cosA_veto"], 
            "DisMuon_DeltaT_vs_CosAlpha_VetoApplied", 
            "Cosmic", 
            PREFIX_DISMUON, 
            OUTPUT_DIR
        )

    print("Done!")