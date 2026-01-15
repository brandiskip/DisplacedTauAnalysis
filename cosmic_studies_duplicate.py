import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hist
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import coffea.nanoevents.methods.vector as vector
import pickle

# --- Plotting Helper Function ---
def save_2d_plot(h, var_name, sample_name, PREFIX, OUTPUT_DIR):
    """Plots a 2D histogram with a logarithmic color scale."""
    if h.sum() == 0:
        print(f"Skipping {var_name} (Histogram is empty)")
        return

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
        # Define histograms for BOTH cases (Same Charge AND Opposite Charge)
        self.output = {
            # Case 1: Opposite Charge
            "dt_vs_cosA_opp_charge": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t = t_{upper} - t_{lower}$ [ns]")
            ),
            "pt_lead_vs_sublead_opp_charge": Hist(
                axis.Regular(100, 0, 200, name="lead_pt", label="Leading Displaced Muon $p_T$ [GeV]"),
                axis.Regular(100, 0, 200, name="sublead_pt", label="Subleading Displaced Muon $p_T$ [GeV]"),
            ),
            
            # Case 2: Same Charge
            "dt_vs_cosA_same_charge": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t = t_{upper} - t_{lower}$ [ns]")
            ),
            "pt_lead_vs_sublead_same_charge": Hist(
                axis.Regular(100, 0, 200, name="lead_pt", label="Leading Displaced Muon $p_T$ [GeV]"),
                axis.Regular(100, 0, 200, name="sublead_pt", label="Subleading Displaced Muon $p_T$ [GeV]"),
            ),

            # Placeholder for veto logic
            "dt_vs_cosA_veto": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t = t_{upper} - t_{lower}$ [ns]")
            ),
        }

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # 1. Attach vector behavior
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

        # 2. Basic Filtering (Must have at least 2 muons)
        mask_2dis = ak.num(events.DisMuon) >= 2
        events = events[mask_2dis]
        
        # 3. Identify Muons (Sorting)
        # Sort by Pt for kinematic plots (Leading vs Subleading)
        sorted_pt = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        lead = sorted_pt[:, 0]
        sublead = sorted_pt[:, 1]

        # Sort by Phi for timing (Upper vs Lower)
        sorted_phi = events.DisMuon[ak.argsort(events.DisMuon.phi, axis=1, ascending=False)]
        upper = sorted_phi[:, 0]
        lower = sorted_phi[:, 1]

        # 4. Calculate Variables for ALL events
        # Delta T
        delta_t = (upper.timeAtIpInOut - lower.timeAtIpInOut)
        
        # Cos Alpha
        dot_product = lead.px * sublead.px + lead.py * sublead.py + lead.pz * sublead.pz
        denominator = lead.p * sublead.p
        cosA = ak.where(denominator != 0, dot_product / denominator, -1000.0)

        # 5. Define Split Masks
        # We do not filter 'events' here. We just create boolean masks.
        # Opposite Charge: q1 * q2 < 0
        mask_opp = (lead.charge * sublead.charge) < 0
        
        # Same Charge: q1 * q2 > 0
        mask_same = (lead.charge * sublead.charge) > 0

        # 6. Fill Histograms using the Masks
        
        # --- CASE 1: Opposite Charge ---
        self.output["dt_vs_cosA_opp_charge"].fill(
            cosA=cosA[mask_opp], 
            dt=delta_t[mask_opp]
        )
        self.output["pt_lead_vs_sublead_opp_charge"].fill(
            lead_pt=lead.pt[mask_opp], 
            sublead_pt=sublead.pt[mask_opp]
        )

        # --- CASE 2: Same Charge ---
        self.output["dt_vs_cosA_same_charge"].fill(
            cosA=cosA[mask_same], 
            dt=delta_t[mask_same]
        )
        self.output["pt_lead_vs_sublead_same_charge"].fill(
            lead_pt=lead.pt[mask_same], 
            sublead_pt=sublead.pt[mask_same]
        )

        return self.output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
    
    # 1. Load Preprocessed Data
    pkl_file_path = "samples/Summer22_CHS_v14_Cosmic/Cosmic_preprocessed.pkl"
    
    with open(pkl_file_path, "rb") as f:
        runnable = pickle.load(f)

    # 2. Run Processor
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

    # 3. Save Plots
    OUTPUT_DIR = "cosmic_muon_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PREFIX = "cosmics_dismuon_"

    # --- Plot Opposite Charge ---
    if "dt_vs_cosA_opp_charge" in out:
        save_2d_plot(out["dt_vs_cosA_opp_charge"], "DisMuon_DeltaT_vs_CosAlpha_OppositeCharge", "Cosmic", PREFIX, OUTPUT_DIR)
        
    if "pt_lead_vs_sublead_opp_charge" in out:
        save_2d_plot(out["pt_lead_vs_sublead_opp_charge"], "DisMuon_Lead_vs_Sublead_pT_OppositeCharge", "Cosmic", PREFIX, OUTPUT_DIR)

    # --- Plot Same Charge ---
    if "dt_vs_cosA_same_charge" in out:
        save_2d_plot(out["dt_vs_cosA_same_charge"], "DisMuon_DeltaT_vs_CosAlpha_SameCharge", "Cosmic", PREFIX, OUTPUT_DIR)
        
    if "pt_lead_vs_sublead_same_charge" in out:
        save_2d_plot(out["pt_lead_vs_sublead_same_charge"], "DisMuon_Lead_vs_Sublead_pT_SameCharge", "Cosmic", PREFIX, OUTPUT_DIR)

    # --- Plot Veto (If used) ---
    if out["dt_vs_cosA_veto"].sum() > 0:
         save_2d_plot(out["dt_vs_cosA_veto"], "DisMuon_DeltaT_vs_CosAlpha_VetoApplied", "Cosmic", PREFIX, OUTPUT_DIR)

    print("Done!")