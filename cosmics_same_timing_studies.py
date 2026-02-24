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
def save_2d_plot(h, var_name, sample_name, PREFIX, OUTPUT_DIR, log_y=False):
    """Plots a 1D or 2D histogram. Added log_y support to fix TypeError."""
    if h.sum() == 0:
        print(f"Skipping {var_name} (Histogram is empty)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 1D Histogram Logic
    if len(h.axes) == 1:
        h.plot1d(ax=ax)
        ax.set_ylabel("Events")
        if log_y:
            ax.set_yscale("log")
        filename = f"{PREFIX}{var_name}.pdf"
    # 2D Histogram Logic
    else:
        is_boolean = h.axes[0].size <= 2
        norm = mcolors.LogNorm(vmin=1) if not is_boolean else None
        h.plot2d(ax=ax, cmap="viridis", norm=norm)
        filename = f"{PREFIX}{var_name}_2D.pdf"
    
    ax.set_title(f"{sample_name}: {var_name}")
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved plot to: {outpath}")

# Update your save_2d_plot function or the loop calling it
def save_2d_plot_with_profile(h, var_name, sample_name, PREFIX, OUTPUT_DIR, log_y=False):
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # plot the standard 2D Histogram (Colormap)
    is_boolean = h.axes[0].size <= 2
    norm = mcolors.LogNorm(vmin=1) if not is_boolean else None
    h.plot2d(ax=ax, cmap="viridis", norm=norm)
    
    # Calculate and Overlay the Profile (Mean & RMS)
    if "phi_reco_vs_gen_lower" in var_name:
        # Get the 2D profile: calculating Mean and Std Dev of Y (reco) vs X (gen)
        # We project onto the X-axis (Gen Phi) to get bins
        profile = h.profile("gen_phi") # This calculates mean of reco_phi per gen_phi bin
        
        # 'profile' is a Hist object holding mean and variance. 
        # However, accessing the raw values directly from the 2D hist is often easier for matplotlib.
        
        # Alternative method using raw values to get RMS specifically:
        x_centers = h.axes[0].centers
        means = []
        rmses = []
        
        # Iterate over x-bins (Gen Phi)
        for i in range(h.axes[0].size):
            # Get 1D slice for this Gen Phi bin
            # Note: hist slicing is inclusive of flow bins usually, but here we iterate valid bins
            y_view = h.view()[i, :] 
            
            # Reconstruct mean and std deviation (RMS) from the 1D slice
            # Weights are simply the counts in the 2D bin
            counts = y_view.value if hasattr(y_view, 'value') else y_view
            y_centers = h.axes[1].centers
            
            total_w = np.sum(counts)
            if total_w > 0:
                avg = np.average(y_centers, weights=counts)
                variance = np.average((y_centers - avg)**2, weights=counts)
                rms = np.sqrt(variance)
                
                means.append(avg)
                rmses.append(rms)
            else:
                means.append(np.nan)
                rmses.append(np.nan)

        # 3. Plot the Profile
        # 'k.' = black dots, err = rms
        ax.errorbar(x_centers, means, yerr=rmses, fmt='k.', capsize=2, 
                    label="Profile (Mean $\pm$ RMS)", zorder=10)
        ax.legend()

    filename = f"{PREFIX}{var_name}_2D_with_RMS.pdf"
    ax.set_title(f"{sample_name}: {var_name}")
    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved plot to: {outpath}")

def save_overlay_plot(h, var_name, PREFIX, OUTPUT_DIR):
    """Overlays different dR cuts from a 2D histogram (Category vs Value)"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Iterate through the categories (the dR cuts)
    for label in h.axes["dr_cut"]:
        # Slice the histogram to get the 1D view for this specific label
        h[{"dr_cut": label}].plot1d(ax=ax, label=label)
    
    ax.legend(title="Matching Threshold")
    ax.set_ylabel("Events")
    ax.set_title(f"Comparison: {var_name}")
    
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_overlay.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved overlay plot to: {outpath}")

def save_dr_overlay(h, var_name, PREFIX, OUTPUT_DIR):
    """Overlays Upper vs Lower dR distributions."""
    if h.sum() == 0:
        print(f"Skipping {var_name} (Histogram is empty)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    
    for label in h.axes["leg"]:
        h[{"leg": label}].plot1d(ax=ax, label=label)
    
    ax.legend(title="Cosmic Leg")
    ax.set_ylabel("Events")
    #ax.set_yscale("log") 
    ax.set_title(f"Comparison: {var_name}")
    
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved dR overlay plot to: {outpath}")

def save_comparison_overlay(h, var_name, PREFIX, OUTPUT_DIR):
    """Overlays the different categories (Upper, Lower, Lower Match) on one plot"""
    if h.sum() == 0:
        print(f"Skipping {var_name} (Empty)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Iterate over the 'cat' axis
    for label in h.axes["cat"]:
        h[{"cat": label}].plot1d(ax=ax, label=label)
    
    ax.legend(title="Leg Category")
    ax.set_ylabel("Events")
    ax.set_title(f"Comparison: {var_name}")
    
    # Optional: Log scale if counts vary wildly
    # ax.set_yscale("log")
    
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_overlay.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved comparison plot to: {outpath}")

def delta_r_mb2(reco_obj, gen_obj):
    """
    Calculates dR using eta_at_mb2 and phi_at_mb2 fields.
    """
    dphi = np.abs(reco_obj.phi_at_mb2 - gen_obj.phi_at_mb2)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = reco_obj.eta_at_mb2 - gen_obj.eta_at_mb2
    return np.sqrt(deta**2 + dphi**2)

# --- The Processor Class ---
class CosmicProcessor(processor.ProcessorABC):
    def __init__(self):
        
        # Helper to create a set of "Duplicate Study" histograms (Kept for reference)
        def make_dup_hists(label):
            return {
                f"deta_{label}": Hist(axis.Regular(100, -0.025, 0.025, name="deta", label=r"$\Delta \eta$")),
                f"dphi_{label}": Hist(axis.Regular(100, -0.01, 0.01, name="dphi", label=r"$\Delta \phi$")),
                f"dpt_{label}": Hist(axis.Regular(100, -1.0, 1.0, name="dpt", label=r"$\Delta p_T$")),
            }

        self.output = {
            # Event Counters
            "n_events_initial": 0,
            "n_duplicates_removed": 0,
            "n_events_final": 0,
            "n_same_side_pairs": 0,
            "n_lead_nan": 0,
            "n_sub_nan": 0,
            "n_lead_under_0": 0,
            "n_sub_under_0": 0,

            # --- MATCHING STUDY PLOTS (NEW) ---
            "dt_double_match": Hist(
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns] (Double Match - Cleaned)"),
            ),
            "dt_single_match": Hist(
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns] (Single Match - Cleaned)"),
            ),
            "dt_zoomed_opp_charge": Hist(
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns]"),
            ),
            "dt_zoomed_same_charge_dup_removed": Hist(
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns]"),
            ),
            "t_upper": Hist(
                axis.Regular(100, -100, 100, name="t_upper", label=r"$t_{upper}$ [ns]")
            ),
            "t_lower": Hist(
                axis.Regular(100, -100, 100, name="t_lower", label=r"$t_{lower}$ [ns]")
            ),
            "t_upper_vs_t_lower_opp_charge": Hist(
                axis.Regular(100, -100, 100, name="t_upper", label=r"$t_{upper}$ [ns]"),
                axis.Regular(100, -100, 100, name="t_lower", label=r"$t_{lower}$ [ns]")
            ),
            "t_upper_vs_t_lower": Hist(
                axis.Regular(100, -100, 100, name="t_upper", label=r"$t_{upper}$ [ns]"),
                axis.Regular(100, -100, 100, name="t_lower", label=r"$t_{lower}$ [ns]")
            ),
            "dt_vs_cosA": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "t_upper_vs_t_lower_cosA_ln0p99": Hist(
                axis.Regular(100, -100, 100, name="t_upper", label=r"$t_{upper}$ [ns]"),
                axis.Regular(100, -100, 100, name="t_lower", label=r"$t_{lower}$ [ns]")
            ),
            "dt_vs_cosA_cosA_ln0p99": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "dt_vs_cosA_same_charge_dup_removed": Hist(
                axis.Regular(50, -1.05, 1.05, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t$ [ns]")
            ),
             "t_upper_vs_t_lower_same_charge_dup_removed": Hist(
                axis.Regular(100, -100, 100, name="t_upper", label=r"$t_{upper}$ [ns]"),
                axis.Regular(100, -100, 100, name="t_lower", label=r"$t_{lower}$ [ns]")
            ),
            "dt_vs_cosA_back_to_back_opp": Hist(
                axis.Regular(20, -1.0, -0.99, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "dt_vs_cosA_back_to_back_same": Hist(
                axis.Regular(20, -1.0, -0.99, name="cosA", label=r"$\cos(\alpha)$"),
                axis.Regular(100, -100, 100, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "n_gen_muons": Hist(
                axis.Regular(5, 0, 5, name="n_gen", label="Number of Gen Muons (status=1)")
            ),
            "dt_double_match_comparison": Hist(
                axis.StrCategory([], name="dr_cut", label="DeltaR Cut", growth=True),
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "dt_single_match_comparison": Hist(
                axis.StrCategory([], name="dr_cut", label="DeltaR Cut", growth=True),
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "dr_distribution_comparison": Hist(
                axis.StrCategory([], name="leg", label="Muon Leg", growth=True),
                axis.Regular(100, 0, 0.4, name="dr", label=r"$\Delta R(\mu_{reco}, \mu_{gen})$")
            ),
            "dxy_vs_dr_upper": Hist(
                axis.Regular(100, 0, 0.5, name="dr", label=r"$\Delta R(\mu_{upper}, \mu_{gen})$"),
                axis.Regular(100, -100, 100, name="dxy", label=r"Muon $d_{xy}$ [cm]")
            ),
            "dxy_vs_dr_lower": Hist(
                axis.Regular(100, 0, 0.5, name="dr", label=r"$\Delta R(\mu_{lower}, \mu_{gen})$"),
                axis.Regular(100, -100, 100, name="dxy", label=r"Muon $d_{xy}$ [cm]")
            ),
            "dr_distribution_comparison_cosA99": Hist(
                axis.StrCategory([], name="leg", label="Muon Leg", growth=True),
                axis.Regular(100, 0, 0.5, name="dr", label=r"$\Delta R(\mu_{reco}, \mu_{gen})$ [cosA < -0.99]")
            ),
            "dxy_vs_dr_upper_cosA99": Hist(
                axis.Regular(100, 0, 0.5, name="dr", label=r"$\Delta R(\mu_{upper}, \mu_{gen})$ [cosA < -0.99]"),
                axis.Regular(100, -100, 100, name="dxy", label=r"Muon $d_{xy}$ [cm]")
            ),
            "dxy_vs_dr_lower_cosA99": Hist(
                axis.Regular(100, 0, 0.5, name="dr", label=r"$\Delta R(\mu_{lower}, \mu_{gen})$ [cosA < -0.99]"),
                axis.Regular(100, -100, 100, name="dxy", label=r"Muon $d_{xy}$ [cm]")
            ),
            "dr_lower_dxy_small": Hist(
                axis.Regular(100, 0, 0.4, name="dr", label=r"$\Delta R(\mu_{lower}, \mu_{gen})$ ($|d_{xy}| < 0.1$)")
            ),
            "dr_lower_dxy_medium": Hist(
                axis.Regular(100, 0, 0.4, name="dr", label=r"$\Delta R(\mu_{lower}, \mu_{gen})$ ($0.1 \leq |d_{xy}| \leq 1.0$)")
            ),
            "dr_lower_dxy_large": Hist(
                axis.Regular(100, 0, 0.4, name="dr", label=r"$\Delta R(\mu_{lower}, \mu_{gen})$ ($|d_{xy}| > 1.0$)")
            ),
            "dt_double_match_comparison_cosA99": Hist(
                axis.StrCategory([], name="dr_cut", label="DeltaR Cut", growth=True),
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns] [cosA < -0.99]")
            ),
            "dt_single_match_comparison_cosA99": Hist(
                axis.StrCategory([], name="dr_cut", label="DeltaR Cut", growth=True),
                axis.Regular(100, -10, 10, name="dt", label=r"$\Delta t$ [ns] [cosA < -0.99]")
            ),
            "phi_reco_vs_gen_lower": Hist(
                axis.Regular(100, -np.pi, np.pi, name="gen_phi", label=r"Gen Muon $\phi$"),
                axis.Regular(100, -np.pi, np.pi, name="reco_phi", label=r"Lower Reco Muon $\phi$")
            ),
            "phi_upper_vs_lower_pos": Hist(
                axis.Regular(100, 0, np.pi, name="phi_lower", label=r"Lower Muon $\phi$ ($\phi > 0$)"),
                axis.Regular(100, 0, np.pi, name="phi_upper", label=r"Upper Muon $\phi$")
            ),
            "eta_reco_vs_gen_lower": Hist(
                axis.Regular(100, -2.5, 2.5, name="gen_eta", label=r"Gen Muon $\eta$"),
                axis.Regular(100, -2.5, 2.5, name="reco_eta", label=r"Lower Reco Muon $\eta$")
            ),
            "phi_reco_vs_gen_best": Hist(
                axis.Regular(100, -np.pi, np.pi, name="gen_phi", label=r"Gen Muon $\phi$"),
                axis.Regular(100, -np.pi, np.pi, name="reco_phi", label=r"Best Match Reco Muon $\phi$")
            ),
            "eta_reco_vs_gen_best": Hist(
                axis.Regular(100, -2.5, 2.5, name="gen_eta", label=r"Gen Muon $\eta$"),
                axis.Regular(100, -2.5, 2.5, name="reco_eta", label=r"Best Match Reco Muon $\eta$")
            ),
            "n_valid_muon_dt_hits": Hist(
                axis.StrCategory([], name="cat", label="Category", growth=True),
                axis.Regular(50, 0, 50, name="val", label="Number of Valid Muon DT Hits")
            ),
            "n_valid_muon_hits": Hist(
                axis.StrCategory([], name="cat", label="Category", growth=True),
                axis.Regular(60, 0, 60, name="val", label="Number of Valid Muon Hits")
            ),
            "n_dt_stations_valid": Hist(
                axis.StrCategory([], name="cat", label="Category", growth=True),
                axis.Regular(6, 0, 6, name="val", label="DT Stations with Valid Hits")
            ),
            "same_side_n_valid_dt_hits": Hist(
                axis.StrCategory([], name="cat", label="Muon Rank (by pT)", growth=True),
                axis.Regular(50, 0, 50, name="val", label="Valid DT Hits (Same-Side Events)")
            ),
            "same_side_n_valid_hits": Hist(
                axis.StrCategory([], name="cat", label="Muon Rank (by pT)", growth=True),
                axis.Regular(60, 0, 60, name="val", label="Valid Muon Hits (Same-Side Events)")
            ),
            "same_side_dt_stations": Hist(
                axis.StrCategory([], name="cat", label="Muon Rank (by pT)", growth=True),
                axis.Regular(6, 0, 6, name="val", label="DT Stations with Valid Hits (Same-Side Events)")
            ),
            "dr_leading_subleading": Hist(
                axis.Regular(100, 0, 5, name="dr", label=r"$\Delta R(\mu_{lead}, \mu_{sublead})$")
            ),
            "same_side_dr_leading_subleading": Hist(
                axis.Regular(100, 0, 5, name="dr", label=r"$\Delta R(\mu_{lead}, \mu_{subsub})$ (Same-Side)")
            ),
            "n_dismuons_after_cuts": Hist(
                axis.Regular(5, 0, 5, name="n", label="Number of DisMuons (After Quality Cuts)")
            ),
            "single_muon_pt": Hist(
                axis.Regular(100, 0, 500, name="pt", label=r"Single Muon $p_T$ [GeV]")
            ),
            "single_muon_eta": Hist(
                axis.Regular(100, -2.5, 2.5, name="eta", label=r"Single Muon $\eta$")
            ),
            "single_muon_phi": Hist(
                axis.Regular(100, -np.pi, np.pi, name="phi", label=r"Single Muon $\phi$")
            ),
            "single_muon_type": Hist(
                axis.StrCategory([], name="type", label="Muon Type", growth=True)
            ),
            "single_lower_dr_gen": Hist(
                axis.Regular(100, 0, 0.4, name="dr", label=r"$\Delta R(\mu_{lower}, \mu_{gen})$ (Single Muon Events)")
            ),
            "single_lower_eta_comp": Hist(
                axis.StrCategory([], name="cat", label="Detector Surface", growth=True),
                axis.Regular(100, -2.5, 2.5, name="val", label=r"Single Lower Muon $\eta$")
            ),
            "single_lower_phi_comp": Hist(
                axis.StrCategory([], name="cat", label="Detector Surface", growth=True),
                axis.Regular(100, -np.pi, np.pi, name="val", label=r"Single Lower Muon $\phi$")
            ),
            "dr_comparison": Hist(
                axis.StrCategory([], name="region", label="dxy Region", growth=True),
                axis.Regular(50, 0, 1, name="dr_cmssw", label=r"CMSSW $\Delta R$"),
                axis.Regular(50, 0, 1, name="dr_manual", label=r"Manual $\Delta R$")
            ),
            "dr_vs_time_dxy_small": Hist(
                axis.Regular(100, -30, 30, name="time", label=r"timeAtIpInOut [ns]"),
                axis.Regular(100, 0, 1.0, name="dr", label=r"$\Delta R$ (Reco, Gen)"),
            ),
            "time_err": Hist(
                axis.Regular(40, 0, 2.5, name="err", label=r"timeAtIpInOutErr [ns]")
            ),
            "time_over_err": Hist(
                axis.Regular(500, -30, 30, name="val", label=r"Time / Uncertainty $(t / \sigma_t)$")
            ),
        }
        # Add duplicate studies keys (Kept in definition)
        self.output.update(make_dup_hists("dups_same_charge"))

    def process(self, events):
        dataset = events.metadata['dataset']
        
        # 1. Attach vector behavior to DisMuons
        events["DisMuon"] = ak.zip(
            {
                "pt": events.DisMuon.pt,
                "ptErr": events.DisMuon.ptErr,
                "eta": events.DisMuon.eta,
                "phi": events.DisMuon.phi,
                "mass": events.DisMuon.mass,
                "charge": events.DisMuon.charge,
                "timeAtIpInOut": events.DisMuon.timeAtIpInOut,
                "timeAtIpInOutErr": events.DisMuon.timeAtIpInOutErr,
                "timeNDof": events.DisMuon.timeNDof,
                "dxy": events.DisMuon.dxy,
                "numberOfValidMuonDTHits": events.DisMuon.numberOfValidMuonDTHits,
                "numberOfValidMuonCSCHits": events.DisMuon.numberOfValidMuonCSCHits,
                "numberOfValidMuonHits": events.DisMuon.numberOfValidMuonHits,
                "dtStationsWithValidHits": events.DisMuon.dtStationsWithValidHits,
                "eta_at_ecal": events.DisMuon.eta_at_ecal,
                "phi_at_ecal": events.DisMuon.phi_at_ecal,
                "eta_at_mb2": events.DisMuon.eta_at_mb2,
                "phi_at_mb2": events.DisMuon.phi_at_mb2,
                "isGlobal": events.DisMuon.isGlobal, 
                "isStandalone": events.DisMuon.isStandalone,
                "staTrackNormChi2": events.DisMuon.staTrackNormChi2,
                "mediumId": events.DisMuon.mediumId,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # 2. Attach vector behavior to GenPart
        events["GenPart"] = ak.zip(
            {
                "pt": events.GenPart.pt,
                "eta": events.GenPart.eta,
                "phi": events.GenPart.phi,
                "mass": events.GenPart.mass,
                "pdgId": events.GenPart.pdgId,
                "status": events.GenPart.status,
                "eta_at_mb2": events.GenPart.eta_at_mb2, 
                "phi_at_mb2": events.GenPart.phi_at_mb2,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        gen_muons = events.GenPart[
            (abs(events.GenPart.pdgId) == 13) & 
            (events.GenPart.status == 1) 
        ]
        
        dis_muons = events.DisMuon[events.DisMuon.pt > 30 &
            (events.DisMuon.mediumId > 0) &
            (events.DisMuon.mediumId < 2)
        ]

        # Standalone Requirements
        sta_hit_base = (dis_muons.numberOfValidMuonCSCHits + dis_muons.numberOfValidMuonDTHits) > 12
        sta_hit_csc0 = ak.where(dis_muons.numberOfValidMuonCSCHits == 0, dis_muons.numberOfValidMuonDTHits > 18, True)
        sta_chi2 = dis_muons.staTrackNormChi2 < 2.5
        sta_pterr = (dis_muons.ptErr / dis_muons.pt) < 1.0
        
        mask_sta = dis_muons.isStandalone & sta_hit_base & sta_hit_csc0 & sta_chi2 & sta_pterr

        # Global Requirements
        glb_hit = (dis_muons.numberOfValidMuonCSCHits + dis_muons.numberOfValidMuonDTHits) > 12
        glb_pterr = (dis_muons.ptErr / dis_muons.pt) < 0.3
        
        mask_glb = dis_muons.isGlobal & glb_hit & glb_pterr

        # Combine and apply to DisMuon collection (Particle-level filter)
        good_muon_mask = mask_sta | mask_glb
        events["DisMuon"] = events.DisMuon[good_muon_mask]

        n_dismuons = ak.num(events.DisMuon)
        self.output["n_dismuons_after_cuts"].fill(n=n_dismuons)

        mask_single = (n_dismuons == 1) & (ak.num(gen_muons) >= 1)
        
        if ak.sum(mask_single) > 0:
            single_muons = events.DisMuon[mask_single][:, 0]
            single_gens = gen_muons[mask_single][:, 0] 
            
            self.output["single_muon_pt"].fill(pt=single_muons.pt)
            self.output["single_muon_eta"].fill(eta=single_muons.eta)
            self.output["single_muon_phi"].fill(phi=single_muons.phi)
            
            is_g = single_muons.isGlobal
            is_s = single_muons.isStandalone
            
            mask_both = is_g & is_s
            mask_g_only = is_g & ~is_s
            mask_s_only = is_s & ~is_g
            
            if ak.sum(mask_both) > 0:
                self.output["single_muon_type"].fill(type=np.full(ak.sum(mask_both), "Global & Standalone"))
            if ak.sum(mask_g_only) > 0:
                self.output["single_muon_type"].fill(type=np.full(ak.sum(mask_g_only), "Global Only"))
            if ak.sum(mask_s_only) > 0:
                self.output["single_muon_type"].fill(type=np.full(ak.sum(mask_s_only), "Standalone Only"))

            mask_upper_single = single_muons.phi > 0
            mask_lower_single = single_muons.phi < 0
            
            lower_muons = single_muons[mask_lower_single]
            lower_gens = single_gens[mask_lower_single]
            
            # Calculate and plot dR for the lower muons
            if len(lower_muons) > 0:
                dr_lower = lower_muons.delta_r(lower_gens)
                self.output["single_lower_dr_gen"].fill(dr=dr_lower)
                abs_dxy_lower = abs(lower_muons.dxy)
                mask_dxy_small  = abs_dxy_lower < 0.1
                mask_dxy_medium = (abs_dxy_lower >= 0.1) & (abs_dxy_lower <= 1.0)
                mask_dxy_large  = abs_dxy_lower > 1.0

                deta = lower_muons.eta - lower_gens.eta
                dphi = lower_muons.phi - lower_gens.phi
                dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                
                dr_manual = np.sqrt(deta**2 + dphi**2)

                if ak.sum(mask_dxy_small) > 0:
                    self.output["dr_comparison"].fill(
                        region="dxy < 0.1",
                        dr_cmssw=dr_lower[mask_dxy_small],
                        dr_manual=dr_manual[mask_dxy_small]
                    )
                if ak.sum(mask_dxy_medium) > 0:
                    self.output["dr_comparison"].fill(
                        region="0.1 <= dxy <= 1.0",
                        dr_cmssw=dr_lower[mask_dxy_medium],
                        dr_manual=dr_manual[mask_dxy_medium]
                    )
                if ak.sum(mask_dxy_large) > 0:
                    self.output["dr_comparison"].fill(
                        region="dxy > 1.0",
                        dr_cmssw=dr_lower[mask_dxy_large],
                        dr_manual=dr_manual[mask_dxy_large]
                    )
                if ak.sum(mask_dxy_small) > 0:
                    self.output["dr_vs_time_dxy_small"].fill(
                        time=lower_muons.timeAtIpInOut[mask_dxy_small],
                        dr=dr_lower[mask_dxy_small]
                    )
                valid_time = lower_muons.timeAtIpInOutErr > 0
                if ak.sum(valid_time) > 0:
                    t_val = lower_muons.timeAtIpInOut[valid_time]
                    t_err = lower_muons.timeAtIpInOutErr[valid_time]

                    self.output["time_err"].fill(err=t_err)

                    # Calculate Pull and Significance
                    t_over_err = t_val / t_err

                    self.output["time_over_err"].fill(val=t_over_err)

                # Fill separate histograms
                self.output["dr_lower_dxy_small"].fill(dr=dr_lower[mask_dxy_small])
                self.output["dr_lower_dxy_medium"].fill(dr=dr_lower[mask_dxy_medium])
                self.output["dr_lower_dxy_large"].fill(dr=dr_lower[mask_dxy_large])

                # Fill Eta
                self.output["single_lower_eta_comp"].fill(cat="IP", val=lower_muons.eta)
                self.output["single_lower_eta_comp"].fill(cat="ECAL", val=lower_muons.eta_at_ecal)
                self.output["single_lower_eta_comp"].fill(cat="MB2", val=lower_muons.eta_at_mb2)
                
                # Fill Phi
                self.output["single_lower_phi_comp"].fill(cat="IP", val=lower_muons.phi)
                self.output["single_lower_phi_comp"].fill(cat="ECAL", val=lower_muons.phi_at_ecal)
                self.output["single_lower_phi_comp"].fill(cat="MB2", val=lower_muons.phi_at_mb2)

        
        # Filter for at least 2 DisMuons AND at least 1 GenMuon (Event-level filter)
        mask_good_event = (ak.num(events.DisMuon) >= 2) & (ak.num(gen_muons) >= 1)
        
        # Apply ONLY the event-level mask here
        events = events[mask_good_event]
        gen_muons = gen_muons[mask_good_event]

        #if ak.any(ak.num(gen_muons) > 1):
            #self.output["n_gen_muons"].fill(n_gen=ak.num(gen_muons))

        self.output["n_events_initial"] += len(events)

        # --- PRE-CLEANING ---
        sorted_pt = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        lead = sorted_pt[:, 0]
        sublead = sorted_pt[:, 1]

        dot_product = lead.px * sublead.px + lead.py * sublead.py + lead.pz * sublead.pz
        denominator = lead.p * sublead.p
        cosA = ak.where(denominator != 0, dot_product / denominator, -1000.0)
        
        deta = lead.eta - sublead.eta
        dphi = lead.delta_phi(sublead)
        dpt = lead.pt - sublead.pt
        
        mask_same = (lead.charge * sublead.charge) > 0
        is_duplicate = mask_same & (abs(deta) < 0.01) & (abs(dphi) < 0.001) & (abs(dpt) < 0.5)
        
        self.output["n_duplicates_removed"] += ak.sum(is_duplicate)

        # Filter all arrays used later
        events = events[~is_duplicate]
        gen_muons = gen_muons[~is_duplicate]
        cosA = cosA[~is_duplicate]
        lead = lead[~is_duplicate]
        sublead = sublead[~is_duplicate]
        
        self.output["n_events_final"] += len(events)

        dr_lead_sub = lead.delta_r(sublead)
        self.output["dr_leading_subleading"].fill(dr=dr_lead_sub)

        # -------------------------------------------------------------------
        # MATCHING STUDY 
        # -------------------------------------------------------------------
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
            self.output["same_side_dr_leading_subleading"].fill(dr=dr_same_side)
            
            # Helper to fill all 3 histograms for a specific rank (slice)
            def fill_same_side_hists(muons, rank_label):
                self.output["same_side_n_valid_dt_hits"].fill(cat=rank_label, val=muons.numberOfValidMuonDTHits)
                self.output["same_side_n_valid_hits"].fill(cat=rank_label, val=muons.numberOfValidMuonHits)
                self.output["same_side_dt_stations"].fill(cat=rank_label, val=muons.dtStationsWithValidHits)

            # 1. Fill Leading Muon (Index 0) - Guaranteed to exist by mask
            fill_same_side_hists(same_side_sorted[:, 0], "Leading pT")

            # 2. Fill Subleading Muon (Index 1) - Guaranteed to exist by mask (n>=2)
            fill_same_side_hists(same_side_sorted[:, 1], "Subleading pT")

            # 3. Fill 3rd Muon (Index 2) - Only if n >= 3
            # We filter for events that actually have a 3rd muon
            mask_has_third = ak.num(same_side_sorted) >= 3
            if ak.any(mask_has_third):
                fill_same_side_hists(same_side_sorted[mask_has_third][:, 2], "3rd Muon")

        '''
        # has at least 3 muons
        has_3_muons = (n_total >= 3)
        if ak.any(has_3_muons):
            print(f"WARNING: Events with >=3 muons")
        
        # EDGE CASE CHECK: 3+ muons ALL on the same side
        # Condition: At least 3 muons, and (0 upper OR 0 lower)
        # Note: We assume 'events.event' exists. If using a custom schema, adjust the field name.
        weird_mask = (n_total >= 3) & ((n_upper == 0) | (n_lower == 0))
        
        if ak.any(weird_mask):
            # Extract event numbers where this happens
            weird_event_ids = events.event[weird_mask]
            print(f"WARNING: Events with >=3 muons all on same side: {weird_event_ids.to_list()}")
        '''    

        
        # Filter Events
        # We only want events that have AT LEAST one Upper AND one Lower candidate.
        # This implicitly removes:
        #   - Events with 2 muons on the same side (e.g., n_upper=2, n_lower=0)
        #   - The "weird" 3+ same-side events identified above
        has_both_legs = (n_upper >= 1) & (n_lower >= 1)
        
        # Apply mask to 'events' and auxiliary arrays to keep them synchronized
        events = events[has_both_legs]
        gen_muons = gen_muons[has_both_legs]
        dis_muons = dis_muons[has_both_legs]

        lead = lead[has_both_legs]          
        sublead = sublead[has_both_legs]
        cosA = cosA[has_both_legs]
        
        # Apply mask to our candidate arrays so we can pick the best ones from the valid events
        upper_candidates = upper_candidates[has_both_legs]
        lower_candidates = lower_candidates[has_both_legs]

        # Resolve Ambiguities (>1 candidate per side)
        # "Keep the one with the highest pt"
        # We sort descending by pt and take the first index ([:, 0])
        upper_candidates = upper_candidates[ak.argsort(upper_candidates.pt, axis=1, ascending=False)]
        lower_candidates = lower_candidates[ak.argsort(lower_candidates.pt, axis=1, ascending=False)]

        # Now we have exactly one Upper and one Lower muon per valid event
        upper = upper_candidates[:, 0]
        lower = lower_candidates[:, 0]

        ndof_quality = (upper.timeNDof > 7) & (lower.timeNDof > 7)

        events = events[ndof_quality]
        gen_muons = gen_muons[ndof_quality]
        dis_muons = dis_muons[ndof_quality]

        lead = lead[ndof_quality]          
        sublead = sublead[ndof_quality]
        cosA = cosA[ndof_quality]

        upper = upper[ndof_quality]
        lower = lower[ndof_quality]

        self.output["t_upper"].fill(
            t_upper=upper.timeAtIpInOut
        )
        self.output["t_lower"].fill(
            t_lower=lower.timeAtIpInOut
        )

        # -------------------------------------------------------------------
        # MATCHING STUDY PLOTS
        # -------------------------------------------------------------------
        
        # Fill the 2D plot 
        self.output["phi_upper_vs_lower_pos"].fill(
            phi_lower=lower.phi, 
            phi_upper=upper.phi
        )
        
        # Calculate Delta T
        delta_t = (upper.timeAtIpInOut - lower.timeAtIpInOut)
        
        # Take the leading GenMuon
        leading_gen = gen_muons[:, 0]

        self.output["dt_vs_cosA"].fill(
            cosA=cosA, 
            dt=delta_t
        )
        self.output["t_upper_vs_t_lower"].fill(
            t_upper=upper.timeAtIpInOut,
            t_lower=lower.timeAtIpInOut
        )

        mask_back_to_back = cosA < -0.99
        self.output["dt_vs_cosA_cosA_ln0p99"].fill(
            cosA=cosA[mask_back_to_back], 
            dt=delta_t[mask_back_to_back]
        )
        self.output["t_upper_vs_t_lower_cosA_ln0p99"].fill(
            t_upper=upper.timeAtIpInOut[mask_back_to_back],
            t_lower=lower.timeAtIpInOut[mask_back_to_back]
        )

        # Fill phi and eta correlation plots for lower muon
        self.output["phi_reco_vs_gen_lower"].fill(
            gen_phi=leading_gen.phi,
            reco_phi=lower.phi
        )
        self.output["eta_reco_vs_gen_lower"].fill(
            gen_eta=leading_gen.eta,
            reco_eta=lower.eta
        )

        # --- dR DISTRIBUTION STUDY ---
        # Calculate dR for each leg against the leading gen muon
        dr_upper = upper.delta_r(leading_gen)
        dr_lower = lower.delta_r(leading_gen)
        dr_lower_mb2 = delta_r_mb2(lower, leading_gen)
        '''
        self.output["n_valid_muon_dt_hits"].fill(cat="Upper", val=upper.numberOfValidMuonDTHits)
        self.output["n_valid_muon_hits"].fill(cat="Upper", val=upper.numberOfValidMuonHits)
        self.output["n_dt_stations_valid"].fill(cat="Upper", val=upper.dtStationsWithValidHits)
        
        # 2. Fill for ALL Lower Muons
        self.output["n_valid_muon_dt_hits"].fill(cat="Lower", val=lower.numberOfValidMuonDTHits)
        self.output["n_valid_muon_hits"].fill(cat="Lower", val=lower.numberOfValidMuonHits)
        self.output["n_dt_stations_valid"].fill(cat="Lower", val=lower.dtStationsWithValidHits)
        
        # 3. Fill for Lower Muons within dR < 0.4 of Gen
        mask_lower_match = dr_lower < 0.4
        self.output["n_valid_muon_dt_hits"].fill(cat="Lower (dR < 0.4)", val=lower.numberOfValidMuonDTHits[mask_lower_match])
        self.output["n_valid_muon_hits"].fill(cat="Lower (dR < 0.4)", val=lower.numberOfValidMuonHits[mask_lower_match])
        self.output["n_dt_stations_valid"].fill(cat="Lower (dR < 0.4)", val=lower.dtStationsWithValidHits[mask_lower_match])

        # Determine which muon is closer to the Gen Muon
        # Returns True if Upper is closer, False if Lower is closer
        mask_upper_is_best = dr_upper < dr_lower

        # Select the Best Muon and its dR using the mask
        best_muon = ak.where(mask_upper_is_best, upper, lower)
        best_dr = ak.where(mask_upper_is_best, dr_upper, dr_lower)

        mask_good_match = best_dr < 0.4

        # Fill the histograms using only the valid best matches
        self.output["phi_reco_vs_gen_best"].fill(
            gen_phi=leading_gen.phi[mask_good_match],
            reco_phi=best_muon.phi[mask_good_match]
        )
        self.output["eta_reco_vs_gen_best"].fill(
            gen_eta=leading_gen.eta[mask_good_match],
            reco_eta=best_muon.eta[mask_good_match]
        )
        
        # Fill the comparison histogram
        self.output["dr_distribution_comparison"].fill(
            leg="Upper Muon",
            dr=dr_upper
        )
        '''
        self.output["dr_distribution_comparison"].fill(
            leg="Lower Muon",
            dr=dr_lower
        )
        '''
        self.output["dxy_vs_dr_upper"].fill(
            dr=dr_upper,
            dxy=upper.dxy
        )
        self.output["dxy_vs_dr_lower"].fill(
            dr=dr_lower,
            dxy=lower.dxy
        )
        
        abs_dxy_lower = abs(lower.dxy)
        mask_dxy_small  = abs_dxy_lower < 0.1
        mask_dxy_medium = (abs_dxy_lower >= 0.1) & (abs_dxy_lower <= 1.0)
        mask_dxy_large  = abs_dxy_lower > 1.0

        # Fill separate histograms
        self.output["dr_lower_dxy_small"].fill(dr=dr_lower[mask_dxy_small])
        self.output["dr_lower_dxy_medium"].fill(dr=dr_lower[mask_dxy_medium])
        self.output["dr_lower_dxy_large"].fill(dr=dr_lower[mask_dxy_large])
        '''

        '''
        # Loop over different dR thresholds to fill categories
        for dr_threshold in [0.4, 0.3, 0.2, 0.1]:
            label = f"dR < {dr_threshold}"
            
            u_match = upper.delta_r(leading_gen) < dr_threshold
            l_match = lower.delta_r(leading_gen) < dr_threshold
            
            mask_double = u_match & l_match
            mask_single = (u_match ^ l_match)

            self.output["dt_double_match_comparison"].fill(
                dr_cut=label, 
                dt=delta_t[mask_double]
            )
            self.output["dt_single_match_comparison"].fill(
                dr_cut=label, 
                dt=delta_t[mask_single]
            )
        '''

        '''
        # Check Delta R matching (0.1)
        upper_matches = upper.delta_r(leading_gen) < 0.1
        lower_matches = lower.delta_r(leading_gen) < 0.1
        
        # Case 1: DOUBLE MATCH
        mask_double = upper_matches & lower_matches
        
        # Case 2: SINGLE MATCH
        mask_single = (upper_matches ^ lower_matches)

        # Fill Histograms
        self.output["dt_double_match"].fill(dt=delta_t[mask_double])
        self.output["dt_single_match"].fill(dt=delta_t[mask_single])
        '''

        '''
        # -------------------------------------------------------------------
        # BACK-TO-BACK STUDY (cosAlpha < -0.99)
        # -------------------------------------------------------------------
        mask_back_to_back = cosA < -0.99
        mask_opp = (lead.charge * sublead.charge) < 0
        mask_opp_clean = mask_opp & mask_back_to_back

        mask_same_charge = (lead.charge * sublead.charge) > 0
        mask_same_back_to_back = mask_same_charge & mask_back_to_back

        self.output["dt_vs_cosA_back_to_back_opp"].fill(
            cosA=cosA[mask_opp_clean], 
            dt=delta_t[mask_opp_clean]
        )
        self.output["dt_vs_cosA_back_to_back_same"].fill(
            cosA=cosA[mask_same_back_to_back], 
            dt=delta_t[mask_same_back_to_back]
        )

        self.output["t_upper_vs_t_lower_opp_charge"].fill(
            t_upper=upper.timeAtIpInOut[mask_opp_clean],
            t_lower=lower.timeAtIpInOut[mask_opp_clean]
        )
        self.output["t_upper_vs_t_lower_same_charge_dup_removed"].fill(
            t_upper=upper.timeAtIpInOut[mask_same_back_to_back],
            t_lower=lower.timeAtIpInOut[mask_same_back_to_back]
        )

        # dR Distribution Comparison (Upper vs Lower)
        self.output["dr_distribution_comparison_cosA99"].fill(
            leg="Upper Muon", dr=dr_upper[mask_back_to_back]
        )
        self.output["dr_distribution_comparison_cosA99"].fill(
            leg="Lower Muon", dr=dr_lower[mask_back_to_back]
        )

        # dxy vs dr (Upper and Lower separated)
        self.output["dxy_vs_dr_upper_cosA99"].fill(
            dr=dr_upper[mask_back_to_back], dxy=upper.dxy[mask_back_to_back]
        )
        self.output["dxy_vs_dr_lower_cosA99"].fill(
            dr=dr_lower[mask_back_to_back], dxy=lower.dxy[mask_back_to_back]
        )

        # dR Threshold Comparison Overlays
        for dr_threshold in [0.4, 0.3, 0.2, 0.1]:
            label = f"dR < {dr_threshold}"
            u_m = dr_upper < dr_threshold
            l_m = dr_lower < dr_threshold
            
            mask_db = u_m & l_m & mask_back_to_back
            mask_sg = (u_m ^ l_m) & mask_back_to_back

            self.output["dt_double_match_comparison_cosA99"].fill(
                dr_cut=label, dt=delta_t[mask_db]
            )
            self.output["dt_single_match_comparison_cosA99"].fill(
                dr_cut=label, dt=delta_t[mask_sg]
            )
        '''
        return self.output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
    
    # Load Preprocessed Data
    pkl_file_path = "samples/Summer22_CHS_v15_Cosmic/Cosmic_preprocessed.pkl"
    
    with open(pkl_file_path, "rb") as f:
        runnable = pickle.load(f)

    # Run Processor
    print("Starting Processor...")
    executor = processor.FuturesExecutor(workers=8)
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
    '''
    # Print the grand totals
    print("\n--- SAME SIDE CHI2 TOTALS ---")
    print(f"Total Same-Side Pairs: {out['n_same_side_pairs']}")
    print(f"Leading NaNs: {out['n_lead_nan']} | Subleading NaNs: {out['n_sub_nan']}")
    print(f"Leading < 0: {out['n_lead_under_0']} | Subleading < 0: {out['n_sub_under_0']}")
    '''
    
    # Save Plots
    OUTPUT_DIR = "cosmic_matching_plots_wSTA_GB_sel"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PREFIX = "matching_cleaned_"

    # Plot ONLY the new Matching plots
    standard_plots = [
        "t_upper",
        "t_lower",
        "dt_vs_cosA",
        "t_upper_vs_t_lower",
        "dt_vs_cosA_cosA_ln0p99",
        "t_upper_vs_t_lower_cosA_ln0p99",
        #"time_err",
        #"time_over_err",
        #"time_pull",           
        #"dr_vs_time_sig",
        #"dr_vs_time_dxy_small",
        #"single_lower_dr_gen",
        #"single_muon_pt",
        #"single_muon_eta",
        #"single_muon_phi",
        #"single_muon_type",
        #"n_dismuons_after_cuts",
        #"same_side_dr_leading_subleading",
        #"dr_leading_subleading",
        #"phi_reco_vs_gen_best",
        #"eta_reco_vs_gen_best",
        #"phi_upper_vs_lower_pos",
        #"dr_lower_dxy_small",
        #"dr_lower_dxy_medium",
        #"dr_lower_dxy_large",
        #"phi_reco_vs_gen_lower",
        #"eta_reco_vs_gen_lower",
        #"n_gen_muons",
        #"dt_vs_cosA_back_to_back_opp",
        #"dt_vs_cosA_back_to_back_same",
        #"t_upper_vs_t_lower_opp_charge",
        #"t_upper_vs_t_lower_same_charge_dup_removed"
        #"dxy_vs_dr_upper", 
        #"dxy_vs_dr_lower",
        #"dxy_vs_dr_upper_cosA99", 
        #"dxy_vs_dr_lower_cosA99" 
    ]

    # 2. Define which plots need the overlay logic
    overlay_plots = [
        #"dt_double_match_comparison",
        #"dt_single_match_comparison",
        #"dt_double_match_comparison_cosA99", 
        #"dt_single_match_comparison_cosA99"
    ]

    dr_comparison_plots = [
        #"dr_distribution_comparison",
        #"dr_distribution_comparison_cosA99"
    ]

    hit_comparison_plots = [
        #"n_valid_muon_dt_hits",
        #"n_valid_muon_hits",
        #"n_dt_stations_valid",
        #"single_lower_eta_comp", 
        #"single_lower_phi_comp"
    ]

    same_side_plots = [
        #"same_side_n_valid_dt_hits",
        #"same_side_n_valid_hits",
        #"same_side_dt_stations",
    ]

    profile_plots = [
        #"phi_reco_vs_gen_lower" 
    ]
    '''
    if "dr_comparison" in out:
        from matplotlib.colors import LogNorm
        dr_hist = out["dr_comparison"]
        
        # Loop through each region in the categorical axis
        for region in dr_hist.axes["region"]:
            # Slice the 3D histogram down to a 2D histogram for this specific region
            h2d = dr_hist[{"region": region}]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Extract edges and values explicitly to prevent pcolormesh TypeErrors
            x_edges = h2d.axes[0].edges
            y_edges = h2d.axes[1].edges
            counts = h2d.values().T
            
            # Plot the 2D colormap
            mesh = ax.pcolormesh(x_edges, y_edges, counts, norm=LogNorm(), cmap="viridis")
            fig.colorbar(mesh, ax=ax, label="Events")
            
            ax.set_xlabel(r"CMSSW $\Delta R$")
            ax.set_ylabel(r"Manual $\Delta R$")
            ax.set_title(f"CMSSW vs Manual dR ({region})")
            
            # Add a perfect agreement diagonal line for visual reference
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label="Perfect Agreement (y=x)")
            ax.legend()
            
            # Clean up the region string for the filename
            safe_region_name = region.replace("<", "lt").replace(">", "gt").replace("=", "eq").replace(" ", "_")
            file_name = f"{OUTPUT_DIR}/{PREFIX}_Cosmic_dr_comparison_{safe_region_name}.pdf"
            
            fig.savefig(file_name, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {file_name}")
            '''
    for key, hist_obj in out.items():
        if isinstance(hist_obj, (int, float)): 
            continue

        # Check against the new list instead of hardcoding the string
        if key in profile_plots:
            save_2d_plot_with_profile(hist_obj, key, "Cosmic", PREFIX, OUTPUT_DIR)    
            
        elif key in standard_plots:
            use_log = (key == "n_gen_muons", "t_upper", "t_lower")
            save_2d_plot(hist_obj, key, "Cosmic", PREFIX, OUTPUT_DIR, log_y=use_log)
            
        elif key in overlay_plots:
            save_overlay_plot(hist_obj, key, PREFIX, OUTPUT_DIR)

        elif key in dr_comparison_plots:
            save_dr_overlay(hist_obj, key, PREFIX, OUTPUT_DIR)

        elif key in hit_comparison_plots:
            save_comparison_overlay(hist_obj, key, PREFIX, OUTPUT_DIR)
            
        elif key in same_side_plots:
            save_comparison_overlay(hist_obj, key, PREFIX, OUTPUT_DIR)

    print("Done!")