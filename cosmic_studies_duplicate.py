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
    
    # Check if this is a boolean/integer ID plot (small axes) to skip LogNorm if needed
    is_boolean = h.axes[0].size <= 2
    norm = mcolors.LogNorm(vmin=1) if not is_boolean else None

    # For 1D histograms (like the new delta plots), use plot1d
    if len(h.axes) == 1:
        h.plot1d(ax=ax)
        ax.set_ylabel("Events")
        filename = f"{PREFIX}{var_name}.pdf"
    else:
        h.plot2d(ax=ax, cmap="viridis", norm=norm)
        filename = f"{PREFIX}{var_name}_2D.pdf"
    
    ax.set_title(f"{sample_name}: {var_name}")
    
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved plot to: {outpath}")

# --- The Processor Class ---
class CosmicProcessor(processor.ProcessorABC):
    def __init__(self):
        
        # Helper to create a set of "Duplicate Study" histograms
        def make_dup_hists(label):
            return {
                # --- EXISTING PLOTS ---
                f"pt_{label}": Hist(
                    axis.Regular(100, 0, 200, name="lead", label="Lead $p_T$ [GeV]"),
                    axis.Regular(100, 0, 200, name="sublead", label="Sublead $p_T$ [GeV]"),
                ),
                f"eta_{label}": Hist(
                    axis.Regular(50, -2.5, 2.5, name="lead", label="Lead $\eta$"),
                    axis.Regular(50, -2.5, 2.5, name="sublead", label="Sublead $\eta$"),
                ),
                f"phi_{label}": Hist(
                    axis.Regular(50, -np.pi, np.pi, name="lead", label="Lead $\phi$"),
                    axis.Regular(50, -np.pi, np.pi, name="sublead", label="Sublead $\phi$"),
                ),
                f"isGlobal_{label}": Hist(
                    axis.Regular(2, 0, 2, name="lead", label="Lead isGlobal (0=No, 1=Yes)"),
                    axis.Regular(2, 0, 2, name="sublead", label="Sublead isGlobal (0=No, 1=Yes)"),
                ),
                f"isStandalone_{label}": Hist(
                    axis.Regular(2, 0, 2, name="lead", label="Lead isStandalone (0=No, 1=Yes)"),
                    axis.Regular(2, 0, 2, name="sublead", label="Sublead isStandalone (0=No, 1=Yes)"),
                ),

                # --- NEW DELTA PLOTS (For Cut Study) ---
                f"deta_{label}": Hist(
                    axis.Regular(100, -0.1, 0.1, name="deta", label=r"$\Delta \eta (\eta_{lead} - \eta_{sub})$"),
                ),
                f"dphi_{label}": Hist(
                    axis.Regular(100, -0.1, 0.1, name="dphi", label=r"$\Delta \phi (\phi_{lead} - \phi_{sub})$"),
                ),
                f"dpt_{label}": Hist(
                    axis.Regular(100, -10, 10, name="dpt", label=r"$\Delta p_T (p_{T,lead} - p_{T,sub})$ [GeV]"),
                ),
            }

        # Define histograms
        self.output = {
            # Event Counters (Initialized to 0)
            "n_same_charge": 0,
            "n_opp_charge": 0,

            # Standard Plots
            "dt_vs_cosA_opp_charge": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "dt_vs_cosA_same_charge": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t$ [ns]")
            ),
        }
        
        # Add duplicate studies for both Same and Opp charge
        self.output.update(make_dup_hists("dups_opp_charge"))
        self.output.update(make_dup_hists("dups_same_charge"))

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # 1. Attach vector behavior AND ID variables
        events["DisMuon"] = ak.zip(
            {
                "pt": events.DisMuon.pt,
                "eta": events.DisMuon.eta,
                "phi": events.DisMuon.phi,
                "mass": events.DisMuon.mass,
                "charge": events.DisMuon.charge,
                "timeAtIpInOut": events.DisMuon.timeAtIpInOut,
                "isGlobal": events.DisMuon.isGlobal, 
                "isStandalone": events.DisMuon.isStandalone,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # 2. Filter for at least 2 muons
        mask_2dis = ak.num(events.DisMuon) >= 2
        events = events[mask_2dis]
        
        # 3. Sort by Pt (Leading vs Subleading)
        sorted_pt = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        lead = sorted_pt[:, 0]
        sublead = sorted_pt[:, 1]

        # 4. Sort by Phi (Upper vs Lower) for Timing
        sorted_phi = events.DisMuon[ak.argsort(events.DisMuon.phi, axis=1, ascending=False)]
        upper = sorted_phi[:, 0]
        lower = sorted_phi[:, 1]

        # 5. Calculate Variables
        delta_t = (upper.timeAtIpInOut - lower.timeAtIpInOut)
        
        dot_product = lead.px * sublead.px + lead.py * sublead.py + lead.pz * sublead.pz
        denominator = lead.p * sublead.p
        cosA = ak.where(denominator != 0, dot_product / denominator, -1000.0)

        # 6. Define Masks (Charge Only)
        mask_opp = (lead.charge * sublead.charge) < 0
        mask_same = (lead.charge * sublead.charge) > 0

        # 7. Count Events (Total Global Count)
        self.output["n_same_charge"] += ak.sum(mask_same)
        self.output["n_opp_charge"] += ak.sum(mask_opp)
        
        # 8. Define Duplicate Condition (cosAlpha > 0.99)
        mask_dups = cosA > 0.99

        # 9. Fill Standard Histograms
        self.output["dt_vs_cosA_opp_charge"].fill(cosA=cosA[mask_opp], dt=delta_t[mask_opp])
        self.output["dt_vs_cosA_same_charge"].fill(cosA=cosA[mask_same], dt=delta_t[mask_same])

        # 10. Fill Duplicate Study Histograms
        def fill_dup_group(label, mask):
            final_mask = mask & mask_dups
            
            # Apply Mask
            l = lead[final_mask]
            s = sublead[final_mask]
            
            # Calculate Deltas
            deta = l.eta - s.eta
            dphi = l.delta_phi(s) # Robust delta phi calculation
            dpt = l.pt - s.pt

            # Fill Histograms
            self.output[f"pt_{label}"].fill(lead=l.pt, sublead=s.pt)
            self.output[f"eta_{label}"].fill(lead=l.eta, sublead=s.eta)
            self.output[f"phi_{label}"].fill(lead=l.phi, sublead=s.phi)
            self.output[f"isGlobal_{label}"].fill(lead=l.isGlobal, sublead=s.isGlobal)
            self.output[f"isStandalone_{label}"].fill(lead=l.isStandalone, sublead=s.isStandalone)
            
            # Fill NEW Delta histograms
            self.output[f"deta_{label}"].fill(deta=deta)
            self.output[f"dphi_{label}"].fill(dphi=dphi)
            self.output[f"dpt_{label}"].fill(dpt=dpt)

        fill_dup_group("dups_opp_charge", mask_opp)
        fill_dup_group("dups_same_charge", mask_same)

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

    # 3. Print Results
    print("\n" + "="*40)
    print("EVENT COUNT RESULTS (Global - No cosA cut)")
    print("="*40)
    print(f"Same Charge Events:     {out['n_same_charge']}")
    print(f"Opposite Charge Events: {out['n_opp_charge']}")
    print("="*40 + "\n")

    # 4. Save Plots
    OUTPUT_DIR = "cosmic_muon_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PREFIX = "cosmics_dismuon_"

    # Plot ONLY the new duplicate study delta plots
    dup_vars = ["deta", "dphi", "dpt"]
    
    for key, hist_obj in out.items():
        if isinstance(hist_obj, int): continue
        
        # Check if the histogram name starts with one of our delta variables
        if any(key.startswith(v) for v in dup_vars):
            save_2d_plot(hist_obj, key, "Cosmic", PREFIX, OUTPUT_DIR)

    # OLD Loop Commented Out
    '''
    for key, hist_obj in out.items():
        if isinstance(hist_obj, int): continue # Skip the counters
        save_2d_plot(hist_obj, key, "Cosmic", PREFIX, OUTPUT_DIR)
    '''

    print("Done!")