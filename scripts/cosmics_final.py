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

def save_2d_profile_overlay(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix=""):
    import numpy as np
    if np.sum(h.values()) == 0:
        return

    # Iterate through each category
    for label in h.axes["cat"]:
        h_slice = h[{"cat": label}]
        if np.sum(h_slice.values()) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))

        # Plot the 2D colormap
        h_slice.plot2d(ax=ax, cmap="viridis")

        # Calculate the profile (Mean Y per X bin)
        x_centers = h_slice.axes[0].centers
        y_centers = h_slice.axes[1].centers
        counts = h_slice.values()

        profile_x = []
        profile_y = []
        profile_yerr = []

        for i in range(len(x_centers)):
            bin_counts = counts[i, :]
            total_in_bin = np.sum(bin_counts)
            if total_in_bin > 0:
                mean_y = np.average(y_centers, weights=bin_counts)
                # Calculate standard error of the mean for accurate error bars
                variance = np.average((y_centers - mean_y)**2, weights=bin_counts)
                std_dev = np.sqrt(variance)
                std_err = std_dev / np.sqrt(total_in_bin)

                profile_x.append(x_centers[i])
                profile_y.append(mean_y)
                profile_yerr.append(std_err)

        # Overlay the profile
        if profile_x:
            ax.errorbar(
                profile_x, profile_y, yerr=profile_yerr,
                fmt='o', color='red', markersize=5, ecolor='red',
                capsize=3, label="Profile (Mean ± StdErr)"
            )
            ax.legend()

        ax.set_title(f"{label}: {var_name} {title_suffix}")

        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{safe_label}_{filename_suffix}.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f"    Saved 2D profile plot to: {outpath}")

def save_simple_2d_plot(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix="", log_z=False):
    import numpy as np
    import matplotlib.colors as mcolors
    if np.sum(h.values()) == 0:
        return

    for label in h.axes["cat"]:
        h_slice = h[{"cat": label}]
        if np.sum(h_slice.values()) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))

        if log_z:
            # Apply logarithmic color scale and set vmin to 1 to avoid log(0) errors
            h_slice.plot2d(ax=ax, cmap="viridis", norm=mcolors.LogNorm(vmin=1))
        else:
            h_slice.plot2d(ax=ax, cmap="viridis")

        ax.set_title(f"{label}: {var_name} {title_suffix}")

        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{safe_label}_{filename_suffix}_2D.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f"    Saved 2D plot to: {outpath}")

# ── Efficiency vs |η_lower| ──────────────────────────────────────────────
def plot_cosA_efficiency_vs_eta(h_pre, h_post, PREFIX, OUTPUT_DIR, title_suffix, filename_suffix):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in h_pre.axes["cat"]:
        pre  = h_pre[{"cat": label}].values()
        post = h_post[{"cat": label}].values()
        centers = h_pre.axes["val"].centers
        eff = np.where(pre > 0, post / pre, np.nan)
        err = np.where(pre > 0, np.sqrt(eff * (1 - np.clip(eff, 0, 1)) / np.clip(pre, 1, None)), np.nan)
        ax.errorbar(centers, eff, yerr=err, fmt='o-', label=label, capsize=3, markersize=4)
    for x, lbl in [(0.9, "Barrel/Overlap"), (1.2, "Overlap/Endcap")]:
        ax.axvline(x, color="gray", linestyle="--", alpha=0.7, label=lbl)
    ax.set_xlabel(r"$|\eta_{\mathrm{lower}}|$")
    ax.set_ylabel(r"Fraction passing $\cos\alpha \geq -0.99$")
    ax.set_title(f"cosα cut efficiency vs lower muon η {title_suffix}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}cosA_eff_vs_eta_lower_{filename_suffix}.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved: {outpath}")

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
            "n_events_3plus_muons_post_cleaning": 0,

            "n_duplicates_removed_cosmic": 0,
            "n_duplicates_removed_nobptx": 0,

            "n_cosmic_events_total": 0,
            "n_nobptx_events_total": 0,

            "cosmic_pre_1": 0, "cosmic_pre_2": 0, "cosmic_pre_3": 0, "cosmic_pre_4": 0, "cosmic_pre_5": 0, "cosmic_pre_gt5": 0, "cosmic_pre_gt10": 0, "cosmic_pre_gt20": 0,
            "cosmic_post_1": 0, "cosmic_post_2": 0, "cosmic_post_3": 0, "cosmic_post_4": 0, "cosmic_post_5": 0, "cosmic_post_gt5": 0, "cosmic_post_gt10": 0, "cosmic_post_gt20": 0,

            "nobptx_pre_1": 0, "nobptx_pre_2": 0, "nobptx_pre_3": 0, "nobptx_pre_4": 0, "nobptx_pre_5": 0, "nobptx_pre_gt5": 0, "nobptx_pre_gt10": 0, "nobptx_pre_gt20": 0,
            "nobptx_post_1": 0, "nobptx_post_2": 0, "nobptx_post_3": 0, "nobptx_post_4": 0, "nobptx_post_5": 0, "nobptx_post_gt5": 0, "nobptx_post_gt10": 0, "nobptx_post_gt20": 0,

            "n_cosmic_same_hemi": 0, "n_cosmic_opp_hemi": 0,
            "n_nobptx_same_hemi": 0, "n_nobptx_opp_hemi": 0,

            "n_cosmic_2mu_pre_cosA": 0,
            "n_cosmic_2mu_post_cosA": 0,
            "n_nobptx_2mu_pre_cosA": 0,
            "n_nobptx_2mu_post_cosA": 0,

            # ── cosA COSMIC REMOVAL STUDY counters ──
            "n_cosA_study_events_cosmic": 0,
            "n_cosA_study_events_nobptx": 0,
            "n_cosA_study_muons_in_cosmic": 0,
            "n_cosA_study_muons_in_nobptx": 0,
            "n_cosA_study_pairs_cosmic": 0,
            "n_cosA_study_pairs_nobptx": 0,
            "n_cosA_study_events_flagged_cosmic": 0,
            "n_cosA_study_events_flagged_nobptx": 0,
            "n_cosA_study_events_surviving_cosmic": 0,
            "n_cosA_study_events_surviving_nobptx": 0,
            "n_cosA_study_muons_surviving_cosmic": 0,
            "n_cosA_study_muons_surviving_nobptx": 0,
            "n_cosA_dt_study_events_surviving_cosmic": 0,
            "n_cosA_dt_study_events_surviving_nobptx": 0,
            "n_cosA_dt_study_events_vetoed_cosmic": 0,
            "n_cosA_dt_study_events_vetoed_nobptx": 0,
            "n_cosA_dt_study_muons_surviving_cosmic": 0,
            "n_cosA_dt_study_muons_surviving_nobptx": 0,

            "delta_time_upper_lower": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True),
                axis.Regular(100, -60, 60, name="val", label=r"$\Delta t$ (Upper - Lower) [ns]")
            ),

            "single_muon_pt": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, 0, 100, name="val", label=r"Single Muon $p_T$ [GeV]")),
            "single_muon_eta": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -2.5, 2.5, name="val", label=r"Single Muon $\eta$")),
            "single_muon_phi": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"Single Muon $\phi$")),
            "single_muon_dxy": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -100, 100, name="val", label=r"Single Muon $d_{xy}$")),
            "single_muon_dz_overlay": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, -300, 300, name="val", label=r"Single Muon $d_z$ [cm]")),
            "single_muon_validDTHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid DT Hits")),
            "single_muon_validCSCHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid CSC Hits")),
            "single_muon_validHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(80, 0, 80, name="val", label="Total Valid Muon Hits")),
            "single_muon_dtStations": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="DT Stations with Valid Hits")),
            "single_muon_timeAtIpInOut": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(25, -60, 60, name="val", label="Time at IP InOut [ns]")),
            "single_muon_timeAtIpInOutErr": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(75, 0, 5, name="val", label="Time at IP InOut Error [ns]")),
            "single_muon_timeNDof": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="timeNDof")),

            "debug_dr3_eta_mb2": Hist(axis.StrCategory([], name="cat", label="Source", growth=True), axis.Regular(110, -105, 5, name="val", label=r"$\eta$ at MB2 ($\Delta R > 3.0$)")),
            "debug_dr3_phi_mb2": Hist(axis.StrCategory([], name="cat", label="Source", growth=True), axis.Regular(110, -105, 5, name="val", label=r"$\phi$ at MB2 ($\Delta R > 3.0$)")),

            "two_muons_cos_alpha": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -1.01, -0.9, name="val", label=r"$\cos\alpha$ (Upper vs Lower)")),

            "same_hemi_dpt": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -50, 50, name="val", label=r"$\Delta p_T$ (Lead - Sublead) [GeV]")),
            "same_hemi_deta": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -5, 5, name="val", label=r"$\Delta\eta$ (Lead - Sublead)")),
            "same_hemi_dphi": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"$\Delta\phi$ (Lead, Sublead)")),
            "same_hemi_deta_mb2": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -5, 5, name="val", label=r"$\Delta\eta$ at MB2 (Lead - Sublead)")),
            "same_hemi_dphi_mb2": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"$\Delta\phi$ at MB2 (Lead, Sublead)")),

            "surviving_cosA": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -1.0, 1.0, name="val", label=r"$\cos\alpha$ (Surviving Cut)")),
            "surviving_dpt": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -50, 50, name="val", label=r"$\Delta p_T$ (Upper - Lower) [GeV]")),
            "surviving_deta": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -5, 5, name="val", label=r"$\Delta\eta$ (Upper - Lower)")),
            "surviving_dphi": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"$\Delta\phi$ (Upper, Lower)")),

            "single_muon_timing_err_ndof7": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (Single, nDof > 7)")),
            "single_muon_timeErr_DT_only": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC == 0)")),
            "single_muon_timeErr_CSC_only": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT == 0, CSC > 0)")),
            "single_muon_timeErr_DT_CSC_both": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC > 0)")),

            "single_muon_timeErr_vs_DTHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Valid DT Hits"),
                axis.Regular(45, 0, 3, name="err", label="Time Error [ns]")
            ),
            "single_muon_timeErr_vs_CSCHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Valid CSC Hits"),
                axis.Regular(45, 0, 3, name="err", label="Time Error [ns]")
            ),
            "single_muon_timeErr_vs_TotalHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Total DT + CSC Hits"),
                axis.Regular(45, 0, 3, name="err", label="Time Error [ns]")
            ),

            "uncut_upper_vs_lower_phi": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(50, 0, np.pi, name="upper", label=r"Upper Muon $\phi$"),
                axis.Regular(50, -np.pi, 0, name="lower", label=r"Lower Muon $\phi$")
            ),
            "uncut_upper_vs_lower_eta": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(100, -2.5, 2.5, name="upper", label=r"Upper Muon $\eta$"),
                axis.Regular(100, -2.5, 2.5, name="lower", label=r"Lower Muon $\eta$")
            ),
            "event_muon_multiplicity": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True),
                axis.Regular(10, 0, 10, name="val", label="Total DisMuons per Event")
            ),
            "event_total_valid_hits": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True),
                axis.Regular(20, 0, 200, name="val", label="Sum of Valid Muon Hits in Event")
            ),
            "event_standalone_fraction": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True),
                axis.Regular(20, 0, 1.05, name="val", label="Fraction of Standalone Muons in Event")
            ),
            "raw_event_muon_multiplicity": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True),
                axis.Regular(20, 0, 20, name="val", label="Raw DisMuons per Event (Pre-Cleaning)")
            ),
            "cosA_vs_eta_lower": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(48, 0, 2.4, name="eta", label=r"$|\eta_{\mathrm{lower}}|$"),
                axis.Regular(100, -1.01, -0.90, name="cosA", label=r"$\cos\alpha$")
            ),
            "cosA_eta_slices": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.StrCategory(["Barrel |η|<0.9", "Crossover 0.9-1.2", "Endcap |η|>1.2"], name="eta_region", label=r"$|\eta|$ Region"),
                axis.Regular(100, -1.001, -0.950, name="val", label=r"$\cos\alpha$")
            ),
            "eta_lower_pre_cosA": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(24, 0, 2.4, name="val", label=r"$|\eta_{\mathrm{lower}}|$")
            ),
            "eta_lower_post_cosA": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(24, 0, 2.4, name="val", label=r"$|\eta_{\mathrm{lower}}|$")
            ),

            # ── cosA COSMIC REMOVAL STUDY histograms ──
            "cosA_study_lead_vs_sub": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(200, -1.01, 1.0, name="val", label=r"$\cos\alpha$ (Lead vs Each Sub-leading)")
            ),
            "cosA_study_surviving_mult": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(10, 0, 10, name="val", label="Muons per Surviving Event (After cosA Veto)")
            ),
            "cosA_study_input_mult": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(20, 0, 20, name="val", label="Muon Multiplicity Entering cosA Study")
            ),
            "cosA_study_surviving_dt": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(100, -60, 60, name="val", label=r"$\Delta t$ (Upper - Lower) [ns]")
            ),
            "cosA_dt_study_surviving_dt": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(100, -60, 60, name="val", label=r"$\Delta t$ (Upper - Lower) [ns] (After cosA + $\Delta t$ cuts)")
            ),

            # Investigate why some events survive cosA and Δt cuts in the MC sample
            # ── Properties of muon pairs surviving cosA + dt cuts ──
            "final_surv_pt": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, 0, 200, name="val", label=r"Muon $p_T$ [GeV]")),
            "final_surv_eta": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -2.5, 2.5, name="val", label=r"Muon $\eta$")),
            "final_surv_phi": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"Muon $\phi$")),
            "final_surv_dxy": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -100, 100, name="val", label=r"Muon $d_{xy}$ [cm]")),
            "final_surv_dz": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -300, 300, name="val", label=r"Muon $d_z$ [cm]")),
            "final_surv_time": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -60, 60, name="val", label="Muon Time [ns]")),
            "final_surv_timeErr": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 5, name="val", label="Muon Time Error [ns]")),
            "final_surv_timeNDof": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(50, 0, 50, name="val", label="Muon timeNDof")),
            "final_surv_DTHits": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 60, name="val", label="Muon Valid DT Hits")),
            "final_surv_CSCHits": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 60, name="val", label="Muon Valid CSC Hits")),
            "final_surv_validHits": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(80, 0, 80, name="val", label="Muon Total Valid Hits")),
            "final_surv_charge": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(3, -1.5, 1.5, name="val", label="Muon Charge")),
            "final_surv_iso": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, 0, 1, name="val", label="Muon pfRelIso03_all")),
            "final_surv_cosA": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -1.0, 0.0, name="val", label=r"$\cos\alpha$ (Upper vs Lower)")),
            "final_surv_dt": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(100, -60, 60, name="val", label=r"$\Delta t$ (Upper - Lower) [ns]")),
            "final_surv_charge_product": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(3, -1.5, 1.5, name="val", label="Charge Product (Upper × Lower)")),
            "final_surv_event_mult": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(10, 0, 10, name="val", label="Total Muons in Surviving Event")),
            "final_surv_cosA_vs_dt": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(50, -1.0, 1.0, name="cosA", label=r"$\cos\alpha$"),
                axis.Regular(50, -60, 60, name="dt", label=r"$\Delta t$ [ns]")
            ),
            "final_surv_upper_eta_vs_lower_eta": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(50, -2.5, 2.5, name="upper", label=r"Upper $\eta$"),
                axis.Regular(50, -2.5, 2.5, name="lower", label=r"Lower $\eta$")
            ),
            "final_surv_upper_pt_vs_lower_pt": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(50, 0, 200, name="upper", label=r"Upper $p_T$ [GeV]"),
                axis.Regular(50, 0, 200, name="lower", label=r"Lower $p_T$ [GeV]")
            ),
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
                "isGlobal": events.DisMuon.isGlobal,
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

        # Ensure we do not crash when searching for GenPart in data
        has_gen = "GenPart" in events.fields

        if has_gen:
            genpart_dict = {
                "pt": events.GenPart.pt,
                "eta": events.GenPart.eta,
                "phi": events.GenPart.phi,
                "mass": events.GenPart.mass,
                "pdgId": events.GenPart.pdgId,
                "status": events.GenPart.status,
                "eta_at_mb2": events.GenPart.eta_at_mb2,
                "phi_at_mb2": events.GenPart.phi_at_mb2,
            }

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
                (events.GenPart.status == 1)
            ]

            gen_muons = gen_muons[(gen_muons.pt > 30) &
                (abs(gen_muons.eta) < 2.4)
            ]

        else:
            gen_muons = None

        dismuon_mask = (events.DisMuon.pt > 30) & \
                        (abs(events.DisMuon.eta) < 2.4) 
        events["DisMuon"] = events.DisMuon[dismuon_mask]

        dis_muons = events.DisMuon
        n_dismuons = ak.num(dis_muons)

        is_cosmic = "Cosmic" in dataset or dataset.startswith("LooseMu") or dataset == "test_cosmics_calib"
        is_nobptx = "NoBPTX" in dataset
        is_signal = not (is_cosmic or is_nobptx)
        ds_label = "MC" if is_cosmic else ("Data" if is_nobptx else "Signal")

        # ==========================================
        # NoBPTX: REMOVE EVENTS WITH GOOD VERTICES
        # (removes out-of-time collisions from residual protons)
        # ==========================================
        if is_nobptx:
            no_vertex_mask = (events.PV.npvsGood == 0)
            events = events[no_vertex_mask]
            if has_gen:
                gen_muons = gen_muons[no_vertex_mask]
            dis_muons = events.DisMuon
            n_dismuons = ak.num(dis_muons)

        '''
        # Track the total raw events explicitly before filtering
        if is_cosmic:
            self.output["n_cosmic_events_total"] += len(events)
        elif is_nobptx:
            self.output["n_nobptx_events_total"] += len(events)

        mask_has_muons = (ak.num(events.DisMuon) > 0)
        events_with_muons = events[mask_has_muons]

        if ak.sum(mask_has_muons) > 0:
            # Sort by pT to find the leading muon
            sorted_muons = events_with_muons.DisMuon[ak.argsort(events_with_muons.DisMuon.pt, axis=1, ascending=False)]
            lead_muon = sorted_muons[:, 0]

            # Require ONLY the leading muon to pass cuts
            lead_quality_mask = (lead_muon.mediumId == True) & \
                                (lead_muon.pfRelIso03_all < 0.18) & \
                                (abs(lead_muon.dxy) > 0.1) &
                                (abs(lead_muon.dxy) < 10)

            # study_events retains all sub-leading shower tracks!
            study_events = events_with_muons[lead_quality_mask]

            if len(study_events) > 0:
                study_muons = study_events.DisMuon
                n_muons_in_event = ak.num(study_muons)
                sum_valid_hits = ak.sum(study_muons.numberOfValidMuonHits, axis=1)
                n_standalone = ak.sum((study_muons.isStandalone == True) & (study_muons.isGlobal == False), axis=1)
                standalone_fraction = n_standalone / n_muons_in_event

                self.output["event_muon_multiplicity"].fill(cat=ds_label, val=n_muons_in_event)
                self.output["event_total_valid_hits"].fill(cat=ds_label, val=sum_valid_hits)
                self.output["event_standalone_fraction"].fill(cat=ds_label, val=standalone_fraction)

                # ==========================================
                # BACKGROUND-ONLY MULTIPLICITY TRACKING
                # ==========================================
                if is_cosmic or is_nobptx:
                    prefix = "cosmic" if is_cosmic else "nobptx"

                    # 1. Fill Pre-Cleaning Counts
                    for m in [1, 2, 3, 4, 5]:
                        self.output[f"{prefix}_pre_{m}"] += ak.sum(n_muons_in_event == m)
                    self.output[f"{prefix}_pre_gt5"] += ak.sum(n_muons_in_event > 5)
                    self.output[f"{prefix}_pre_gt10"] += ak.sum(n_muons_in_event > 10)
                    self.output[f"{prefix}_pre_gt20"] += ak.sum(n_muons_in_event > 20)

                    # 2. Simulate Duplicate Track Removal
                    sorted_study = study_muons[ak.argsort(study_muons.pt, axis=1, ascending=False)]
                    l_muon = sorted_study[:, 0]
                    deta = sorted_study.eta - l_muon.eta
                    dphi = sorted_study.delta_phi(l_muon)
                    dpt = sorted_study.pt - l_muon.pt
                    mask_sc = (sorted_study.charge * l_muon.charge) > 0

                    is_dup = mask_sc & (abs(deta) < 0.01) & (abs(dphi) < 0.001) & (abs(dpt) < 0.5)
                    is_dup = is_dup & (ak.local_index(sorted_study, axis=1) > 0)
                    cleaned_study = sorted_study[~is_dup]

                    # 3. Fill Post-Cleaning Counts
                    n_post = ak.num(cleaned_study)
                    for m in [1, 2, 3, 4, 5]:
                        self.output[f"{prefix}_post_{m}"] += ak.sum(n_post == m)
                    self.output[f"{prefix}_post_gt5"] += ak.sum(n_post > 5)
                    self.output[f"{prefix}_post_gt10"] += ak.sum(n_post > 10)
                    self.output[f"{prefix}_post_gt20"] += ak.sum(n_post > 20)
        '''
        
        # ==========================================
        # SIGNAL-ONLY EVENT FILTERING
        # ==========================================
        if is_signal and has_gen:
            charged_sel = events.Jet.constituents.pf.charge != 0
            dxy = ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = -1))
            dxy = ak.fill_none(dxy, -999)
            events["Jet"] = ak.with_field(events.Jet, dxy, where = "dxy")
            jets = events.Jet[
                (abs(events.Jet.eta) < 2.4) &
                (events.Jet.pt > 32) &
                (events.Jet.neHEF < 0.99) &
                (events.Jet.neEmEF < 0.9) &
                ((events.Jet.chMultiplicity + events.Jet.neMultiplicity) > 1) &
                (events.Jet.chMultiplicity > 0) &
                (events.Jet.muEF < 0.1) &
                (events.Jet.chEmEF < 0.8) &
                (events.Jet.disTauTag_score1 > 0.9) &
                (abs(events.Jet.dxy) > 0.02)
            ]

            good_MET = (events.PFMET.pt > 105)

            # Require exactly one GenMuon and exactly one Reco Jet (post-kinematic cuts)
            signal_mask = (ak.num(gen_muons) == 1) & (ak.num(jets) == 1) & good_MET

            # Apply mask strictly to the signal events
            events = events[signal_mask]
            gen_muons = gen_muons[signal_mask]

            # Redefine DisMuons based on the newly filtered events array
            dis_muons = events.DisMuon
            n_dismuons = ak.num(dis_muons)

        # ==========================================
        # GLOBAL DUPLICATE TRACK REMOVAL
        # (applied before single/multi muon split so cleaned counts feed both)
        # ==========================================
        mask_has_muons_g = (n_dismuons >= 1)
        if ak.sum(mask_has_muons_g) > 0:
            events = events[mask_has_muons_g]
            if has_gen:
                gen_muons = gen_muons[mask_has_muons_g]
            dis_muons = dis_muons[mask_has_muons_g]

            sorted_all = dis_muons[ak.argsort(dis_muons.pt, axis=1, ascending=False)]
            lead_all = sorted_all[:, 0]

            lead_quality_mask_g = (lead_all.mediumId == True) & (lead_all.pfRelIso03_all < 0.18)
            events = events[lead_quality_mask_g]
            if has_gen:
                gen_muons = gen_muons[lead_quality_mask_g]
            sorted_all = sorted_all[lead_quality_mask_g]
            dis_muons = sorted_all  

            if len(events) > 0:
                lead_for_dup = sorted_all[:, 0]
                deta_g = sorted_all.eta - lead_for_dup.eta
                dphi_g = sorted_all.delta_phi(lead_for_dup)
                dpt_g = sorted_all.pt - lead_for_dup.pt
                mask_sc_g = (sorted_all.charge * lead_for_dup.charge) > 0

                is_duplicate_g = mask_sc_g & (abs(deta_g) < 0.01) & (abs(dphi_g) < 0.001) & (abs(dpt_g) < 0.5)
                is_duplicate_g = is_duplicate_g & (ak.local_index(sorted_all, axis=1) > 0)

                if is_cosmic:
                    self.output["n_duplicates_removed_cosmic"] += ak.sum(is_duplicate_g)
                elif is_nobptx:
                    self.output["n_duplicates_removed_nobptx"] += ak.sum(is_duplicate_g)

                dis_muons = sorted_all[~is_duplicate_g]

        n_dismuons = ak.num(dis_muons)

        # ==========================================
        # EXACTLY TWO MUONS (UNCUT) LOGIC
        # ==========================================
        mask_exactly_two_uncut = (n_dismuons == 2)
        if ak.sum(mask_exactly_two_uncut) > 0:
            two_muons_uncut = dis_muons[mask_exactly_two_uncut]

            # Sort by phi so [0] is the highest and [1] is the lowest
            sorted_by_phi = two_muons_uncut[ak.argsort(two_muons_uncut.phi, axis=1, ascending=False)]
            upper_candidates = sorted_by_phi[:, 0]
            lower_candidates = sorted_by_phi[:, 1]

            # Enforce that the upper candidate is strictly positive and the lower is strictly negative
            mask_opposite_hemispheres = (upper_candidates.phi > 0) & (lower_candidates.phi < 0)

            upper_final = upper_candidates[mask_opposite_hemispheres]
            lower_final = lower_candidates[mask_opposite_hemispheres]

            self.output["uncut_upper_vs_lower_phi"].fill(
                cat=ds_label,
                upper=upper_final.phi,
                lower=lower_final.phi
            )
            self.output["uncut_upper_vs_lower_eta"].fill(
                cat=ds_label,
                upper=upper_final.eta,
                lower=lower_final.eta
            )

        # ==========================================
        # cosA COSMIC REMOVAL STUDY (>=2 MUONS, COSMIC/NOBPTX ONLY)
        # ──────────────────────────────────────────
        # For every event with >=2 DisMuons after duplicate removal:
        #   0. Require timeNDof > 7 on ALL DisMuons first
        #   1. Sort by pT, apply mediumId + iso < 0.18 to lead only
        #   2. Compute cosA between lead and EACH sub-leading muon
        #   3. If cosA < -0.99 for a pair -> flag BOTH lead & that
        #      sub-leading as cosmic and remove them
        #   4. Track how many muons are removed and how many remain
        # No HLT trigger applied.
        # Uses independent variables (cs_ prefix) so it does not
        # interfere with the existing multi-muon logic below.
        # ==========================================
        if (is_cosmic or is_nobptx):
            # ── Apply timeNDof > 7 to ALL DisMuons before any cosA/dt cuts ──
            cs_all   = dis_muons[dis_muons.timeNDof > 7]
            n_cs_all = ak.num(cs_all)
            cs_mask  = (n_cs_all >= 2)

            if ak.sum(cs_mask) > 0:
                cs_muons = cs_all[cs_mask]

                # Sort by pT descending
                cs_sorted = cs_muons[ak.argsort(cs_muons.pt, axis=1, ascending=False)]
                cs_lead = cs_sorted[:, 0]

                # Apply mediumId + isolation to the leading muon only
                cs_lead_pass = (
                    (cs_lead.mediumId == True) &
                    (cs_lead.pfRelIso03_all < 0.18)
                )
                cs_sorted = cs_sorted[cs_lead_pass]

                prefix_cs = "cosmic" if is_cosmic else "nobptx"

                if len(cs_sorted) > 0:
                    cs_lead = cs_sorted[:, 0]
                    cs_sub  = cs_sorted[:, 1:]   # all sub-leading muons

                    self.output[f"n_cosA_study_events_{prefix_cs}"] += len(cs_sorted)
                    self.output[f"n_cosA_study_muons_in_{prefix_cs}"] += int(ak.sum(ak.num(cs_sorted)))

                    # ── cosA between lead and each sub-leading ──
                    dot_cs = (cs_lead.px * cs_sub.px +
                              cs_lead.py * cs_sub.py +
                              cs_lead.pz * cs_sub.pz)
                    denom_cs = cs_lead.p * cs_sub.p
                    cosA_cs = ak.where(denom_cs != 0, dot_cs / denom_cs, -1000.0)

                    # Fill the full cosA distribution
                    self.output["cosA_study_lead_vs_sub"].fill(
                        cat=ds_label, val=ak.flatten(cosA_cs)
                    )

                    # Fill input multiplicity
                    self.output["cosA_study_input_mult"].fill(
                        cat=ds_label, val=ak.num(cs_sorted)
                    )

                    # ── Flag entire event if ANY pair has cosA < -0.99 ──
                    event_has_cosmic = ak.any(cosA_cs < -0.99, axis=1)

                    n_pairs_checked = int(ak.sum(ak.num(cs_sub)))
                    n_flagged_events = int(ak.sum(event_has_cosmic))
                    n_surviving_events = int(ak.sum(~event_has_cosmic))

                    self.output[f"n_cosA_study_pairs_{prefix_cs}"] += n_pairs_checked
                    self.output[f"n_cosA_study_events_flagged_{prefix_cs}"] += n_flagged_events
                    self.output[f"n_cosA_study_events_surviving_{prefix_cs}"] += n_surviving_events

                    surviving_muons = cs_sorted[~event_has_cosmic]
                    self.output[f"n_cosA_study_muons_surviving_{prefix_cs}"] += int(
                        ak.sum(ak.num(surviving_muons))
                    )

                    if n_surviving_events > 0:
                        self.output["cosA_study_surviving_mult"].fill(
                            cat=ds_label,
                            val=ak.num(surviving_muons)
                        )

                        # ── Delta t for surviving events: upper - lower hemisphere ──
                        surv_upper_all = surviving_muons[surviving_muons.phi > 0]
                        surv_lower_all = surviving_muons[surviving_muons.phi < 0]

                        # Require at least one in each hemisphere
                        surv_has_both = (ak.num(surv_upper_all) >= 1) & (ak.num(surv_lower_all) >= 1)

                        if ak.sum(surv_has_both) > 0:
                            surv_upper_both = surv_upper_all[surv_has_both]
                            surv_lower_both = surv_lower_all[surv_has_both]

                            # Pick the highest pT in each hemisphere
                            surv_upper_sorted = surv_upper_both[ak.argsort(surv_upper_both.pt, axis=1, ascending=False)]
                            surv_lower_sorted = surv_lower_both[ak.argsort(surv_lower_both.pt, axis=1, ascending=False)]

                            surv_upper = surv_upper_sorted[:, 0]
                            surv_lower = surv_lower_sorted[:, 0]

                            surv_dt = surv_upper.timeAtIpInOut - surv_lower.timeAtIpInOut
                            self.output["cosA_study_surviving_dt"].fill(
                                cat=ds_label,
                                val=surv_dt
                            )

                            # ── Apply dt cut: veto entire event if dt < -20 ──
                            # timeNDof > 7 already required on all muons above, so
                            # the per-leg ndof check here is redundant.
                            dt_pass = (surv_dt >= -20)
                            dt_fail = ~dt_pass

                            self.output[f"n_cosA_dt_study_events_vetoed_{prefix_cs}"] += int(ak.sum(dt_fail))
                            self.output[f"n_cosA_dt_study_events_surviving_{prefix_cs}"] += int(ak.sum(dt_pass))

                            # Get the full surviving_muons for events that had both hemispheres
                            surviving_both_hemi = surviving_muons[surv_has_both]

                            # Keep only events passing the dt cut
                            final_surviving = surviving_both_hemi[dt_pass]
                            self.output[f"n_cosA_dt_study_muons_surviving_{prefix_cs}"] += int(ak.sum(ak.num(final_surviving)))
                            '''
                            if ak.sum(dt_pass) > 0 and is_cosmic:
                                fs_upper = surv_upper[dt_pass]
                                fs_lower = surv_lower[dt_pass]
                                fs_dt = surv_dt[dt_pass]

                                # Recompute cosA for the upper-lower pair
                                fs_dot = fs_upper.px * fs_lower.px + fs_upper.py * fs_lower.py + fs_upper.pz * fs_lower.pz
                                fs_denom = fs_upper.p * fs_lower.p
                                fs_cosA = ak.where(fs_denom != 0, fs_dot / fs_denom, -1000.0)

                                self.output["cosA_dt_study_surviving_dt"].fill(cat=ds_label, val=fs_dt)

                                # Fill overlaid upper vs lower (same histogram, different category)
                                ul = ds_label  # e.g. "MC" or "Data"
                                self.output["final_surv_pt"].fill(cat=f"{ul} Upper", val=fs_upper.pt)
                                self.output["final_surv_pt"].fill(cat=f"{ul} Lower", val=fs_lower.pt)
                                self.output["final_surv_eta"].fill(cat=f"{ul} Upper", val=fs_upper.eta)
                                self.output["final_surv_eta"].fill(cat=f"{ul} Lower", val=fs_lower.eta)
                                self.output["final_surv_phi"].fill(cat=f"{ul} Upper", val=fs_upper.phi)
                                self.output["final_surv_phi"].fill(cat=f"{ul} Lower", val=fs_lower.phi)
                                self.output["final_surv_dxy"].fill(cat=f"{ul} Upper", val=fs_upper.dxy)
                                self.output["final_surv_dxy"].fill(cat=f"{ul} Lower", val=fs_lower.dxy)
                                self.output["final_surv_dz"].fill(cat=f"{ul} Upper", val=fs_upper.dz)
                                self.output["final_surv_dz"].fill(cat=f"{ul} Lower", val=fs_lower.dz)
                                self.output["final_surv_time"].fill(cat=f"{ul} Upper", val=fs_upper.timeAtIpInOut)
                                self.output["final_surv_time"].fill(cat=f"{ul} Lower", val=fs_lower.timeAtIpInOut)
                                self.output["final_surv_timeErr"].fill(cat=f"{ul} Upper", val=fs_upper.timeAtIpInOutErr)
                                self.output["final_surv_timeErr"].fill(cat=f"{ul} Lower", val=fs_lower.timeAtIpInOutErr)
                                self.output["final_surv_timeNDof"].fill(cat=f"{ul} Upper", val=fs_upper.timeNDof)
                                self.output["final_surv_timeNDof"].fill(cat=f"{ul} Lower", val=fs_lower.timeNDof)
                                self.output["final_surv_DTHits"].fill(cat=f"{ul} Upper", val=fs_upper.numberOfValidMuonDTHits)
                                self.output["final_surv_DTHits"].fill(cat=f"{ul} Lower", val=fs_lower.numberOfValidMuonDTHits)
                                self.output["final_surv_CSCHits"].fill(cat=f"{ul} Upper", val=fs_upper.numberOfValidMuonCSCHits)
                                self.output["final_surv_CSCHits"].fill(cat=f"{ul} Lower", val=fs_lower.numberOfValidMuonCSCHits)
                                self.output["final_surv_validHits"].fill(cat=f"{ul} Upper", val=fs_upper.numberOfValidMuonHits)
                                self.output["final_surv_validHits"].fill(cat=f"{ul} Lower", val=fs_lower.numberOfValidMuonHits)
                                self.output["final_surv_charge"].fill(cat=f"{ul} Upper", val=fs_upper.charge)
                                self.output["final_surv_charge"].fill(cat=f"{ul} Lower", val=fs_lower.charge)
                                self.output["final_surv_iso"].fill(cat=f"{ul} Upper", val=fs_upper.pfRelIso03_all)
                                self.output["final_surv_iso"].fill(cat=f"{ul} Lower", val=fs_lower.pfRelIso03_all)

                                # Per-pair quantities (no upper/lower split)
                                self.output["final_surv_cosA"].fill(cat=ds_label, val=fs_cosA)
                                self.output["final_surv_dt"].fill(cat=ds_label, val=fs_dt)
                                self.output["final_surv_charge_product"].fill(cat=ds_label, val=fs_upper.charge * fs_lower.charge)
                                self.output["final_surv_event_mult"].fill(cat=ds_label, val=ak.num(final_surviving))
                                self.output["final_surv_cosA_vs_dt"].fill(cat=ds_label, cosA=fs_cosA, dt=fs_dt)
                                self.output["final_surv_upper_eta_vs_lower_eta"].fill(cat=ds_label, upper=fs_upper.eta, lower=fs_lower.eta)
                                self.output["final_surv_upper_pt_vs_lower_pt"].fill(cat=ds_label, upper=fs_upper.pt, lower=fs_lower.pt)
                            '''
                            
                            # Fill dt for events surviving BOTH cuts
                            if ak.sum(dt_pass) > 0:
                                self.output["cosA_dt_study_surviving_dt"].fill(
                                    cat=ds_label,
                                    val=surv_dt[dt_pass]
                                )

                            # ── Debug: log any events surviving both cuts with >2 muons ──
                            if ak.sum(dt_pass) > 0:
                                final_gt2 = final_surviving[ak.num(final_surviving) > 2]
                                if len(final_gt2) > 0:
                                    n_upper_gt2 = ak.sum(final_gt2.phi > 0, axis=1)
                                    n_lower_gt2 = ak.sum(final_gt2.phi < 0, axis=1)
                                    with open("gt2_surviving_debug.txt", "a") as dbg:
                                        dbg.write(f"[{ds_label}] Events with >2 muons surviving cosA + dt cuts: {len(final_gt2)}\n")
                                        dbg.write(f"  Upper counts: {ak.to_list(n_upper_gt2)}\n")
                                        dbg.write(f"  Lower counts: {ak.to_list(n_lower_gt2)}\n")
                                        dbg.write(f"  timeAtIpInOut: {ak.to_list(final_gt2.timeAtIpInOut)}\n\n")
                            
                        # Also count events that had NO opposite hemisphere pair as vetoed by dt
                        # (they can't form an upper-lower pair, so we can't compute dt)
                        surv_no_both = surviving_muons[~surv_has_both]
                        if len(surv_no_both) > 0:
                            # These events survive cosA but have no upper-lower pair for dt cut
                            # Counting them as surviving the dt cut since it doesn't apply
                            self.output[f"n_cosA_dt_study_events_surviving_{prefix_cs}"] += len(surv_no_both)
                            self.output[f"n_cosA_dt_study_muons_surviving_{prefix_cs}"] += int(ak.sum(ak.num(surv_no_both)))
        '''
        if is_signal:
            sig_mask = (n_dismuons >= 2)

            if ak.sum(sig_mask) > 0:
                sig_muons = dis_muons[sig_mask]

                # Sort by pT, apply lead-muon quality (mediumId + iso + dxy window)
                sig_sorted = sig_muons[ak.argsort(sig_muons.pt, axis=1, ascending=False)]
                sig_lead = sig_sorted[:, 0]
                sig_lead_pass = (
                    (sig_lead.mediumId == True) &
                    (sig_lead.pfRelIso03_all < 0.18) &
                    (abs(sig_lead.dxy) > 0.1) &
                    (abs(sig_lead.dxy) < 10)
                )
                sig_sorted = sig_sorted[sig_lead_pass]

                if len(sig_sorted) > 0:
                    sig_lead = sig_sorted[:, 0]
                    sig_sub  = sig_sorted[:, 1:]   # all sub-leading muons

                    # cosA between lead and each sub-leading muon
                    dot_sig = (sig_lead.px * sig_sub.px +
                               sig_lead.py * sig_sub.py +
                               sig_lead.pz * sig_sub.pz)
                    denom_sig = sig_lead.p * sig_sub.p
                    cosA_sig = ak.where(denom_sig != 0, dot_sig / denom_sig, -1000.0)

                    # Veto entire event if ANY pair has cosA < -0.99
                    event_has_cosmic_sig = ak.any(cosA_sig < -0.99, axis=1)
                    surviving_sig = sig_sorted[~event_has_cosmic_sig]

                    if len(surviving_sig) > 0:
                        # Δt for surviving events: upper - lower hemisphere
                        sig_upper_all = surviving_sig[surviving_sig.phi > 0]
                        sig_lower_all = surviving_sig[surviving_sig.phi < 0]

                        sig_has_both = (ak.num(sig_upper_all) >= 1) & (ak.num(sig_lower_all) >= 1)

                        if ak.sum(sig_has_both) > 0:
                            sig_upper_both = sig_upper_all[sig_has_both]
                            sig_lower_both = sig_lower_all[sig_has_both]

                            # Highest pT in each hemisphere
                            sig_upper = sig_upper_both[ak.argsort(sig_upper_both.pt, axis=1, ascending=False)][:, 0]
                            sig_lower = sig_lower_both[ak.argsort(sig_lower_both.pt, axis=1, ascending=False)][:, 0]

                            sig_dt = sig_upper.timeAtIpInOut - sig_lower.timeAtIpInOut
                            self.output["cosA_study_surviving_dt"].fill(
                                cat=ds_label,   # "Signal"
                                val=sig_dt
                            )                
        '''
        '''
        # ==========================================
        # SINGLE MUON LOGIC
        # ==========================================
        if has_gen:
            mask_single = (n_dismuons == 1) & (ak.num(gen_muons) >= 1)
        else:
            mask_single = (n_dismuons == 1)

        if ak.sum(mask_single) > 0:
            single_muons = dis_muons[mask_single][:, 0]
            if has_gen:
                valid_gen_muons = gen_muons[mask_single]

            mask_medium = (single_muons.mediumId == True)
            mask_iso = (single_muons.pfRelIso03_all < 0.18)

            single_muons = single_muons[mask_medium & mask_iso]
            if has_gen:
                valid_gen_muons = valid_gen_muons[mask_medium & mask_iso]

            if is_cosmic or is_nobptx:
                prefix = "Cosmic" if is_cosmic else "NoBPTX"

                mask_upper_single = single_muons.phi > 0
                mask_lower_single = single_muons.phi < 0
                upper_singles = single_muons[mask_upper_single]
                lower_singles = single_muons[mask_lower_single]

                if len(upper_singles) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat=f"Upper {prefix}", val=upper_singles.dz)
                    self.output["single_muon_pt"].fill(cat=f"Upper {prefix}", val=upper_singles.pt)
                    self.output["single_muon_eta"].fill(cat=f"Upper {prefix}", val=upper_singles.eta)
                    self.output["single_muon_phi"].fill(cat=f"Upper {prefix}", val=upper_singles.phi)
                    self.output["single_muon_dxy"].fill(cat=f"Upper {prefix}", val=upper_singles.dxy)
                    self.output["single_muon_validDTHits"].fill(cat=f"Upper {prefix}", val=upper_singles.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat=f"Upper {prefix}", val=upper_singles.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat=f"Upper {prefix}", val=upper_singles.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat=f"Upper {prefix}", val=upper_singles.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat=f"Upper {prefix}", val=upper_singles.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat=f"Upper {prefix}", val=upper_singles.timeAtIpInOutErr)
                    self.output["single_muon_timeNDof"].fill(cat=f"Upper {prefix}", val=upper_singles.timeNDof)
                    self.output["single_muon_timing_err_ndof7"].fill(cat=f"Upper {prefix}", val=upper_singles[upper_singles.timeNDof > 7].timeAtIpInOutErr)

                    up_dt_only = (upper_singles.numberOfValidMuonDTHits > 0) & (upper_singles.numberOfValidMuonCSCHits == 0)
                    up_csc_only = (upper_singles.numberOfValidMuonDTHits == 0) & (upper_singles.numberOfValidMuonCSCHits > 0)
                    up_both = (upper_singles.numberOfValidMuonDTHits > 0) & (upper_singles.numberOfValidMuonCSCHits > 0)

                    self.output["single_muon_timeErr_DT_only"].fill(cat=f"Upper {prefix}", val=upper_singles[up_dt_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_CSC_only"].fill(cat=f"Upper {prefix}", val=upper_singles[up_csc_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_DT_CSC_both"].fill(cat=f"Upper {prefix}", val=upper_singles[up_both].timeAtIpInOutErr)

                    self.output["single_muon_timeErr_vs_DTHits"].fill(cat=f"Upper {prefix}", hits=upper_singles.numberOfValidMuonDTHits[up_dt_only], err=upper_singles.timeAtIpInOutErr[up_dt_only])
                    self.output["single_muon_timeErr_vs_CSCHits"].fill(cat=f"Upper {prefix}", hits=upper_singles.numberOfValidMuonCSCHits[up_csc_only], err=upper_singles.timeAtIpInOutErr[up_csc_only])
                    self.output["single_muon_timeErr_vs_TotalHits"].fill(cat=f"Upper {prefix}", hits=(upper_singles.numberOfValidMuonDTHits[up_both] + upper_singles.numberOfValidMuonCSCHits[up_both]), err=upper_singles.timeAtIpInOutErr[up_both])

                if len(lower_singles) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat=f"Lower {prefix}", val=lower_singles.dz)
                    self.output["single_muon_pt"].fill(cat=f"Lower {prefix}", val=lower_singles.pt)
                    self.output["single_muon_eta"].fill(cat=f"Lower {prefix}", val=lower_singles.eta)
                    self.output["single_muon_phi"].fill(cat=f"Lower {prefix}", val=lower_singles.phi)
                    self.output["single_muon_dxy"].fill(cat=f"Lower {prefix}", val=lower_singles.dxy)
                    self.output["single_muon_validDTHits"].fill(cat=f"Lower {prefix}", val=lower_singles.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat=f"Lower {prefix}", val=lower_singles.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat=f"Lower {prefix}", val=lower_singles.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat=f"Lower {prefix}", val=lower_singles.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat=f"Lower {prefix}", val=lower_singles.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat=f"Lower {prefix}", val=lower_singles.timeAtIpInOutErr)
                    self.output["single_muon_timeNDof"].fill(cat=f"Lower {prefix}", val=lower_singles.timeNDof)
                    self.output["single_muon_timing_err_ndof7"].fill(cat=f"Lower {prefix}", val=lower_singles[lower_singles.timeNDof > 7].timeAtIpInOutErr)

                    dn_dt_only = (lower_singles.numberOfValidMuonDTHits > 0) & (lower_singles.numberOfValidMuonCSCHits == 0)
                    dn_csc_only = (lower_singles.numberOfValidMuonDTHits == 0) & (lower_singles.numberOfValidMuonCSCHits > 0)
                    dn_both = (lower_singles.numberOfValidMuonDTHits > 0) & (lower_singles.numberOfValidMuonCSCHits > 0)

                    self.output["single_muon_timeErr_DT_only"].fill(cat=f"Lower {prefix}", val=lower_singles[dn_dt_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_CSC_only"].fill(cat=f"Lower {prefix}", val=lower_singles[dn_csc_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_DT_CSC_both"].fill(cat=f"Lower {prefix}", val=lower_singles[dn_both].timeAtIpInOutErr)

                    self.output["single_muon_timeErr_vs_DTHits"].fill(cat=f"Lower {prefix}", hits=lower_singles.numberOfValidMuonDTHits[dn_dt_only], err=lower_singles.timeAtIpInOutErr[dn_dt_only])
                    self.output["single_muon_timeErr_vs_CSCHits"].fill(cat=f"Lower {prefix}", hits=lower_singles.numberOfValidMuonCSCHits[dn_csc_only], err=lower_singles.timeAtIpInOutErr[dn_csc_only])
                    self.output["single_muon_timeErr_vs_TotalHits"].fill(cat=f"Lower {prefix}", hits=(lower_singles.numberOfValidMuonDTHits[dn_both] + lower_singles.numberOfValidMuonCSCHits[dn_both]), err=lower_singles.timeAtIpInOutErr[dn_both])

            elif has_gen: # Signal logic
                dr_mb2_prop_array = delta_r_mb2_prop(single_muons, valid_gen_muons)
                min_dr_mb2_prop = ak.min(dr_mb2_prop_array, axis=1)
                mask_dr_prop = ak.fill_none(min_dr_mb2_prop < 0.4, False)
                matched_muons_prop = single_muons[mask_dr_prop]

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
                    self.output["single_muon_timeNDof"].fill(cat="Signal (Propagated)", val=matched_muons_prop.timeNDof)
                    self.output["single_muon_timing_err_ndof7"].fill(cat="Signal (Propagated)", val=matched_muons_prop[matched_muons_prop.timeNDof > 7].timeAtIpInOutErr)

                    sig_dt_only = (matched_muons_prop.numberOfValidMuonDTHits > 0) & (matched_muons_prop.numberOfValidMuonCSCHits == 0)
                    sig_csc_only = (matched_muons_prop.numberOfValidMuonDTHits == 0) & (matched_muons_prop.numberOfValidMuonCSCHits > 0)
                    sig_both = (matched_muons_prop.numberOfValidMuonDTHits > 0) & (matched_muons_prop.numberOfValidMuonCSCHits > 0)

                    self.output["single_muon_timeErr_DT_only"].fill(cat="Signal (Propagated)", val=matched_muons_prop[sig_dt_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_CSC_only"].fill(cat="Signal (Propagated)", val=matched_muons_prop[sig_csc_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_DT_CSC_both"].fill(cat="Signal (Propagated)", val=matched_muons_prop[sig_both].timeAtIpInOutErr)

                    self.output["single_muon_timeErr_vs_DTHits"].fill(cat="Signal (Propagated)", hits=matched_muons_prop.numberOfValidMuonDTHits[sig_dt_only], err=matched_muons_prop.timeAtIpInOutErr[sig_dt_only])
                    self.output["single_muon_timeErr_vs_CSCHits"].fill(cat="Signal (Propagated)", hits=matched_muons_prop.numberOfValidMuonCSCHits[sig_csc_only], err=matched_muons_prop.timeAtIpInOutErr[sig_csc_only])
                    self.output["single_muon_timeErr_vs_TotalHits"].fill(cat="Signal (Propagated)", hits=(matched_muons_prop.numberOfValidMuonDTHits[sig_both] + matched_muons_prop.numberOfValidMuonCSCHits[sig_both]), err=matched_muons_prop.timeAtIpInOutErr[sig_both])
        '''
        '''
        # ==========================================
        # MULTIPLE MUON LOGIC
        # ==========================================
        if has_gen:
            mask_multiple_disMuon_event = (n_dismuons >= 2) & (ak.num(gen_muons) >= 1)
        else:
            mask_multiple_disMuon_event = (n_dismuons >= 2)

        if ak.sum(mask_multiple_disMuon_event) > 0:
            events = events[mask_multiple_disMuon_event]
            if has_gen:
                gen_muons = gen_muons[mask_multiple_disMuon_event]
            dis_muons = dis_muons[mask_multiple_disMuon_event]
            n_dismuons = ak.num(dis_muons)

            # --- Apply cuts to Leading pT Muon First ---
            sorted_muons_temp = dis_muons[ak.argsort(dis_muons.pt, axis=1, ascending=False)]
            lead_muon_eval = sorted_muons_temp[:, 0]

            lead_quality_mask = (lead_muon_eval.mediumId == True) & (lead_muon_eval.pfRelIso03_all < 0.18)

            # Remove the event entirely if the leading muon fails the cuts
            events = events[lead_quality_mask]
            if has_gen:
                gen_muons = gen_muons[lead_quality_mask]
            sorted_muons = sorted_muons_temp[lead_quality_mask]

            if len(events) > 0:
                mask_exactly_two = (ak.num(sorted_muons) == 2)

                events = events[mask_exactly_two]
                if has_gen:
                    gen_muons = gen_muons[mask_exactly_two]
                final_muons = sorted_muons[mask_exactly_two]

                if ak.sum(mask_exactly_two) > 0:
                    mu1 = final_muons[:, 0]
                    mu2 = final_muons[:, 1]

                    mask_same_hemi = (mu1.phi * mu2.phi) > 0
                    mask_opp_hemi = (mu1.phi * mu2.phi) < 0

                    if is_cosmic or is_nobptx:
                        if is_cosmic:
                            self.output["n_cosmic_same_hemi"] += ak.sum(mask_same_hemi)
                            self.output["n_cosmic_opp_hemi"] += ak.sum(mask_opp_hemi)
                        elif is_nobptx:
                            self.output["n_nobptx_same_hemi"] += ak.sum(mask_same_hemi)
                            self.output["n_nobptx_opp_hemi"] += ak.sum(mask_opp_hemi)

                        same_hemi_muons = final_muons[mask_same_hemi]
                        if ak.sum(mask_same_hemi) > 0:
                            mu1_sh = same_hemi_muons[:, 0]
                            mu2_sh = same_hemi_muons[:, 1]

                            dpt_sh = mu1_sh.pt - mu2_sh.pt
                            deta_sh = mu1_sh.eta - mu2_sh.eta
                            dphi_sh = mu1_sh.delta_phi(mu2_sh)

                            # MB2 variables are bare floats, so we manually calculate Delta and wrap to [-pi, pi]
                            deta_mb2_sh = mu1_sh.eta_at_mb2 - mu2_sh.eta_at_mb2

                            dphi_mb2_sh = mu1_sh.phi_at_mb2 - mu2_sh.phi_at_mb2
                            dphi_mb2_sh = ak.where(dphi_mb2_sh > np.pi, dphi_mb2_sh - 2*np.pi, dphi_mb2_sh)
                            dphi_mb2_sh = ak.where(dphi_mb2_sh <= -np.pi, dphi_mb2_sh + 2*np.pi, dphi_mb2_sh)

                            self.output["same_hemi_dpt"].fill(cat=ds_label, val=dpt_sh)
                            self.output["same_hemi_deta"].fill(cat=ds_label, val=deta_sh)
                            self.output["same_hemi_dphi"].fill(cat=ds_label, val=dphi_sh)
                            self.output["same_hemi_deta_mb2"].fill(cat=ds_label, val=deta_mb2_sh)
                            self.output["same_hemi_dphi_mb2"].fill(cat=ds_label, val=dphi_mb2_sh)

                    upper_candidates = final_muons[final_muons.phi > 0]
                    lower_candidates = final_muons[final_muons.phi < 0]

                    n_upper = ak.num(upper_candidates)
                    n_lower = ak.num(lower_candidates)

                    has_both_legs = (n_upper == 1) & (n_lower == 1)

                    events = events[has_both_legs]
                    if has_gen:
                        gen_muons = gen_muons[has_both_legs]
                    upper_candidates = upper_candidates[has_both_legs]
                    lower_candidates = lower_candidates[has_both_legs]

                    if ak.sum(has_both_legs) > 0 and (is_cosmic or is_nobptx):
                        upper = upper_candidates[:, 0]
                        lower = lower_candidates[:, 0]

                        dot_product = upper.px * lower.px + upper.py * lower.py + upper.pz * lower.pz
                        denominator = upper.p * lower.p
                        cosA = ak.where(denominator != 0, dot_product / denominator, -1000.0)

                        self.output["two_muons_cos_alpha"].fill(
                                cat=ds_label,
                                val=cosA
                            )

                        dt = upper.timeAtIpInOut - lower.timeAtIpInOut
                        self.output["delta_time_upper_lower"].fill(
                            cat=ds_label,
                            val=dt
                        )

                        mask_pass_cosA = (cosA >= -0.99)

                        if is_cosmic:
                            self.output["n_cosmic_2mu_pre_cosA"] += len(cosA)
                            self.output["n_cosmic_2mu_post_cosA"] += ak.sum(mask_pass_cosA)
                        elif is_nobptx:
                            self.output["n_nobptx_2mu_pre_cosA"] += len(cosA)
                            self.output["n_nobptx_2mu_post_cosA"] += ak.sum(mask_pass_cosA)

                        # cosα by |η| region (CMS Run 3 muon system):
                        #   Barrel DT-only : |η| < 0.9
                        #   Overlap DT+CSC : 0.9 ≤ |η| ≤ 1.2
                        #   Endcap CSC-only: |η| > 1.2
                        # Classify each pair by the leg with larger |η|.
                        abs_eta_lower = np.abs(lower.eta)
                        self.output["eta_lower_pre_cosA"].fill(cat=ds_label, val=abs_eta_lower)
                        self.output["eta_lower_post_cosA"].fill(cat=ds_label, val=abs_eta_lower[mask_pass_cosA])

                        self.output["cosA_vs_eta_lower"].fill(
                            cat=ds_label,
                            eta=abs_eta_lower,
                            cosA=cosA
                        )

                        eta_regions = [
                            (abs_eta_lower < 0.9,                                 "Barrel |η|<0.9"),
                            ((abs_eta_lower >= 0.9) & (abs_eta_lower <= 1.2),     "Crossover 0.9-1.2"),
                            (abs_eta_lower > 1.2,                                 "Endcap |η|>1.2"),
                        ]
                        for region_mask, region_label in eta_regions:
                            if ak.sum(region_mask) > 0:
                                self.output["cosA_eta_slices"].fill(
                                    cat=ds_label,
                                    eta_region=region_label,
                                    val=cosA[region_mask]
                                )

                        if ak.sum(mask_pass_cosA) > 0:
                            surv_cosA = cosA[mask_pass_cosA]
                            surv_upper = upper[mask_pass_cosA]
                            surv_lower = lower[mask_pass_cosA]

                            self.output["surviving_cosA"].fill(cat=ds_label, val=surv_cosA)

                            # Calculate Deltas for the surviving pairs
                            dpt = surv_upper.pt - surv_lower.pt
                            deta = surv_upper.eta - surv_lower.eta
                            # awkward array vector methods give us delta_phi directly
                            dphi = surv_upper.delta_phi(surv_lower)

                            self.output["surviving_dpt"].fill(cat=ds_label, val=dpt)
                            self.output["surviving_deta"].fill(cat=ds_label, val=deta)
                            self.output["surviving_dphi"].fill(cat=ds_label, val=dphi)
                        '''
        return self.output

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':

    # Load the preprocessed Cosmic pickle file
    cosmic_pkl = "scripts/samples/Summer22_CHS_collisionCalib_v19_Cosmic/Cosmic_CollisionCalib_preprocessed.pkl"
    print(f"Loading preprocessed Cosmics from {cosmic_pkl}...")
    with open(cosmic_pkl, "rb") as f:
        combined_runnable = pickle.load(f)

    # --- COMMENTED OUT NOBPTX LOAD ---
    nobptx_pkl = "samples/Summer22_CHS_v19_Cosmic/NoBPTX_preprocessed.pkl"
    print(f"Loading preprocessed NoBPTX from {nobptx_pkl}...")
    with open(nobptx_pkl, "rb") as f:
        nobptx_runnable = pickle.load(f)

    # Merge the dictionaries
    combined_runnable.update(nobptx_runnable)

    # --- ADDED BACK SIGNAL LOAD ---
    signal_pkl = "samples/Signal/Stau_300_100mm_preprocessed.pkl"
    print(f"Loading preprocessed Signal from {signal_pkl}...")
    with open(signal_pkl, "rb") as f:
        signal_runnable = pickle.load(f)

    # Merge the dictionaries
    combined_runnable.update(signal_runnable)

    # Run the Processor
    print("Starting Processor...")
    executor = processor.FuturesExecutor(workers=8)
    runner = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
        skipbadfiles=True,
    )

    out = runner(
        combined_runnable,
        treename="Events",
        processor_instance=SingleMuonProcessor(),
    )

    print("="*80)
    print(" cosA COSMIC REMOVAL STUDY (>=2 MUONS, FULL EVENT VETO)")
    print("="*80)
    print(" For events with >=2 DisMuons after duplicate removal (Cosmic/NoBPTX only):")
    print("   1. All DisMuons pass pT > 30 GeV, |eta| < 2.4")
    print("   2. Sort by pT; apply mediumId + pfRelIso03_all < 0.18 to LEAD only")
    print("   3. Compute cosA between lead and each sub-leading muon")
    print("   4. If ANY pair has cosA < -0.99: DISCARD THE ENTIRE EVENT")
    print("   5. Separate survivors into upper/lower hemisphere by phi (then sort by pT)")
    print("   6. If dt (upper - lower) < -20 ns: DISCARD THE ENTIRE EVENT")
    print("-"*80)

    for ds_name, prefix in [("COSMICS MC", "cosmic"), ("NoBPTX DATA", "nobptx")]:
        n_evt     = out[f'n_cosA_study_events_{prefix}']
        n_mu_in   = out[f'n_cosA_study_muons_in_{prefix}']
        n_pairs   = out[f'n_cosA_study_pairs_{prefix}']
        n_flagged = out[f'n_cosA_study_events_flagged_{prefix}']
        n_surv    = out[f'n_cosA_study_events_surviving_{prefix}']
        n_mu_surv = out[f'n_cosA_study_muons_surviving_{prefix}']

        n_dt_vetoed  = out[f'n_cosA_dt_study_events_vetoed_{prefix}']
        n_dt_surv    = out[f'n_cosA_dt_study_events_surviving_{prefix}']
        n_dt_mu_surv = out[f'n_cosA_dt_study_muons_surviving_{prefix}']

        print(f"\n  {ds_name}:")
        print(f"    Events entering study (>=2 muons, lead passes cuts): {n_evt}")
        print(f"    Total muons entering:                                {n_mu_in}")
        print(f"    Lead vs sub-leading pairs checked:                   {n_pairs}")
        print(f"    ── After cosA cut (cosA < -0.99 → veto event) ──")
        print(f"    Events vetoed by cosA:                               {n_flagged}")
        print(f"    Events surviving cosA:                               {n_surv}")
        if n_evt > 0:
            print(f"    cosA survival rate: {n_surv}/{n_evt} = {(n_surv/n_evt)*100:.2f}%")
        print(f"    Muons in cosA-surviving events:                      {n_mu_surv}")
        print(f"    ── After dt cut (dt < -20 ns → veto event) ──")
        print(f"    Events vetoed by dt:                                 {n_dt_vetoed}")
        print(f"    Events surviving cosA + dt:                          {n_dt_surv}")
        if n_evt > 0:
            print(f"    Combined survival rate: {n_dt_surv}/{n_evt} = {(n_dt_surv/n_evt)*100:.2f}%")
            print(f"    Combined veto rate:     {(n_flagged + n_dt_vetoed)}/{n_evt} = {((n_flagged + n_dt_vetoed)/n_evt)*100:.2f}%")
        print(f"    Muons in final surviving events:                     {n_dt_mu_surv}")
        if n_mu_in > 0:
            print(f"    Final muon survival rate: {n_dt_mu_surv}/{n_mu_in} = {(n_dt_mu_surv/n_mu_in)*100:.2f}%")

    print("="*80 + "\n")

    # ── cosA study plots go in their own folder ──
    COSA_OUTPUT_DIR = "cosA_study_plots"
    os.makedirs(COSA_OUTPUT_DIR, exist_ok=True)

    COSA_PREFIX = "cosA_study_"
    COSA_TITLE = "(Cosmics MC vs NoBPTX Data)"
    COSA_FILE = "MC_vs_Data"

    cosA_overlay_plots = [
        "cosA_study_lead_vs_sub",
        "cosA_study_surviving_mult",
        "cosA_study_input_mult",
        "cosA_study_surviving_dt",
        "cosA_dt_study_surviving_dt",
        #"final_surv_pt",
        #"final_surv_eta",
        #"final_surv_phi",
        #"final_surv_dxy",
        #"final_surv_dz",
        #"final_surv_time",
        #"final_surv_timeErr",
        #"final_surv_timeNDof",
        #"final_surv_DTHits",
        #"final_surv_CSCHits",
        #"final_surv_validHits",
        #"final_surv_charge",
        #"final_surv_iso",
        #"final_surv_cosA",
        #"final_surv_dt",
        #"final_surv_charge_product",
        #"final_surv_event_mult",
    ]

    cosA_2d_plots = [
        #"final_surv_cosA_vs_dt",
        #"final_surv_upper_eta_vs_lower_eta",
        #"final_surv_upper_pt_vs_lower_pt",
    ]

    OUTPUT_DIR = "single_muon_signal_vs_cosmic_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Apply formatting to the titles and filenames
    PREFIX = "single_muon_"
    TITLE_MODIFIER = "(300 GeV 100 mm vs Cosmics)"
    FILE_MODIFIER = "Stau_300_100mm_overlay"

    overlay_plots = [
        #"delta_time_upper_lower",
        #"event_muon_multiplicity",
        #"event_total_valid_hits",
        #"event_standalone_fraction",
        #"same_hemi_dpt",
        #"same_hemi_deta",
        #"same_hemi_dphi",
        #"same_hemi_deta_mb2",
        #"same_hemi_dphi_mb2",
        #"surviving_cosA",
        #"surviving_dpt",
        #"surviving_deta",
        #"surviving_dphi",
        #"single_muon_dz_overlay",
        #"single_muon_validDTHits",
        #"single_muon_validCSCHits",
        #"single_muon_validHits",
        #"single_muon_dtStations",
        #"single_muon_pt",
        #"single_muon_eta",
        #"single_muon_phi",
        #"single_muon_dxy",
        #"single_muon_timeNDof",
        #"single_muon_timeAtIpInOut",
        #"single_muon_timeAtIpInOutErr",
        #"single_muon_timing_err_ndof7",
        #"single_muon_timeErr_DT_only",
        #"single_muon_timeErr_CSC_only",
        #"single_muon_timeErr_DT_CSC_both",
        #"two_muons_cos_alpha",
        #"raw_event_muon_multiplicity",
    ]

    profile_plots = [
        #"single_muon_timeErr_vs_DTHits",
        #"single_muon_timeErr_vs_CSCHits",
        #"single_muon_timeErr_vs_TotalHits"
    ]

    simple_2d_plots = [
        #"uncut_upper_vs_lower_phi",
        #"uncut_upper_vs_lower_eta"
    ]

    for key, hist_obj in out.items():
        if isinstance(hist_obj, (int, float)):
            continue

        if key in cosA_overlay_plots:
            save_comparison_overlay(hist_obj, key, COSA_PREFIX, COSA_OUTPUT_DIR,
                                    title_suffix=COSA_TITLE, filename_suffix=COSA_FILE,
                                    normalize=True)
        elif key in cosA_2d_plots:
            save_simple_2d_plot(hist_obj, key, COSA_PREFIX, COSA_OUTPUT_DIR,
                                title_suffix=COSA_TITLE, filename_suffix=COSA_FILE,
                                log_z=False)

        elif key in overlay_plots:
            save_comparison_overlay(hist_obj, key, PREFIX, OUTPUT_DIR, title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER, normalize=True)

        elif key in profile_plots:
            save_2d_profile_overlay(hist_obj, key, PREFIX, OUTPUT_DIR, title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER)

        elif key in simple_2d_plots:
            save_simple_2d_plot(hist_obj, key, PREFIX, OUTPUT_DIR, title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER, log_z=True)
    '''
    plot_cosA_efficiency_vs_eta(
        out["eta_lower_pre_cosA"], out["eta_lower_post_cosA"],
        PREFIX, OUTPUT_DIR, TITLE_MODIFIER, FILE_MODIFIER
    )

    # ── 2D profile: zoom y-axis to where events actually are ─────────────────
    save_2d_profile_overlay(
        out["cosA_vs_eta_lower"], "cosA_vs_eta_lower",
        PREFIX, OUTPUT_DIR,
        title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER
    )

    # ── 1D cosα slices: log scale + zoomed range ─────────────────────────────
    for region_label in ["Barrel |η|<0.9", "Crossover 0.9-1.2", "Endcap |η|>1.2"]:
        h_region = out["cosA_eta_slices"][{"eta_region": region_label}]
        safe_region = (region_label
            .replace("|η|", "eta")
            .replace("|", "")
            .replace(" ", "_")
            .replace("<", "lt")
            .replace(">", "gt")
            .replace(".", "p")
        )
        save_comparison_overlay(
            h_region, f"cosA_{safe_region}",
            PREFIX, OUTPUT_DIR,
            title_suffix=f"{region_label} {TITLE_MODIFIER}",
            filename_suffix=FILE_MODIFIER,
            normalize=True,
            log_y=True
        )
    '''
    print("Done!")