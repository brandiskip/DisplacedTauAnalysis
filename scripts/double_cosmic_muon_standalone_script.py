import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import mplhep as hep
hep.style.use("CMS")

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

def save_comparison_overlay(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix="",
                            log_y=False, normalize=True, ratio_ylim=(0.5, 1.5)):
    if np.sum(h.values()) == 0:
        print(f"Skipping {var_name} (Empty)")
        return

    all_cats = list(h.axes["cat"])

    for hemisphere in ["Upper", "Lower"]:
        mc_label   = f"{hemisphere} MC"
        data_label = f"{hemisphere} Data"

        if mc_label not in all_cats or data_label not in all_cats:
            continue

        h_mc   = h[{"cat": mc_label}]
        h_data = h[{"cat": data_label}]

        mc_raw     = h_mc.values()
        data_raw   = h_data.values()
        mc_total   = np.sum(mc_raw)
        data_total = np.sum(data_raw)

        if mc_total == 0 or data_total == 0:
            continue

        if normalize:
            h_mc_plot   = h_mc   * (1.0 / mc_total)
            h_data_plot = h_data * (1.0 / data_total)
        else:
            h_mc_plot   = h_mc
            h_data_plot = h_data

        mc_vals     = h_mc_plot.values()
        data_vals   = h_data_plot.values()
        bin_centers = h_mc_plot.axes[0].centers
        bin_edges   = h_mc_plot.axes[0].edges
        bin_width   = bin_edges[1] - bin_edges[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mc_vals > 0, data_vals / mc_vals, np.nan)
            ratio_err = np.where(
                data_raw > 0,
                ratio * np.sqrt(1.0 / data_raw),
                np.nan,
            )

        # ── Figure layout ────────────────────────────────────────────
        fig = plt.figure(figsize=(8, 9))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main  = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        # ── Main panel ───────────────────────────────────────────────
        ax_main.stairs(
            mc_vals, bin_edges,
            color="#5790fc", linewidth=1.5,
            fill=True, alpha=0.4, label=f"MC  (N={int(mc_total)})",
        )
        ax_main.stairs(
            mc_vals, bin_edges,
            color="#5790fc", linewidth=1.5,
        )
        ax_main.errorbar(
            bin_centers, data_vals,
            yerr=np.where(data_raw > 0, data_vals / np.sqrt(data_raw), 0),
            fmt="+", color="black", markersize=6, elinewidth=1,
            capsize=2, label=f"Data  (N={int(data_total)})",
        )

        ylabel = f"Fraction of Events / {bin_width:.2g}" if normalize else f"Events / {bin_width:.2g}"
        ax_main.set_ylabel(ylabel)

        # Title sits above the axes frame; CMS label sits inside the top-left of the axes.
        # These two never overlap.
        ax_main.set_title(f"{hemisphere} Muons — {var_name}", fontsize=10, pad=6)
        hep.cms.label(data=True, label="Private Work", com=13.6, ax=ax_main, fontsize=13)

        ax_main.legend(frameon=False, fontsize=9)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Suppress the bottom y-tick label so it doesn't bleed into the ratio panel.
        ax_main.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="lower"))

        if log_y:
            ax_main.set_yscale("log")

        # ── Ratio panel ──────────────────────────────────────────────
        ax_ratio.errorbar(
            bin_centers, ratio, yerr=ratio_err,
            fmt="o", color="black", markersize=3,
            elinewidth=1, capsize=2,
        )
        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax_ratio.set_ylabel("Data / MC", fontsize=12)
        ax_ratio.set_ylim(*ratio_ylim)
        ax_ratio.set_xlabel(h_mc_plot.axes[0].label)
        ax_ratio.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax_ratio.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        ax_ratio.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

        safe_hem = hemisphere.lower()
        outpath  = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{safe_hem}_{filename_suffix}.pdf")
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {hemisphere} plot → {outpath}")

def save_2d_profile_overlay(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix=""):
    if np.sum(h.values()) == 0:
        return

    for label in h.axes["cat"]:
        h_slice = h[{"cat": label}]
        if np.sum(h_slice.values()) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))
        h_slice.plot2d(ax=ax, cmap="viridis")

        x_centers = h_slice.axes[0].centers
        y_centers = h_slice.axes[1].centers
        counts = h_slice.values()

        profile_x, profile_y, profile_yerr = [], [], []
        for i in range(len(x_centers)):
            bin_counts = counts[i, :]
            total_in_bin = np.sum(bin_counts)
            if total_in_bin > 0:
                mean_y = np.average(y_centers, weights=bin_counts)
                variance = np.average((y_centers - mean_y) ** 2, weights=bin_counts)
                profile_x.append(x_centers[i])
                profile_y.append(mean_y)
                profile_yerr.append(np.sqrt(variance) / np.sqrt(total_in_bin))

        if profile_x:
            ax.errorbar(
                profile_x, profile_y, yerr=profile_yerr,
                fmt="o", color="red", markersize=5, ecolor="red",
                capsize=3, label="Profile (Mean ± StdErr)",
            )
            ax.legend()

        ax.set_title(f"{label}: {var_name} {title_suffix}")
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{safe_label}_{filename_suffix}.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f"    Saved 2D profile plot → {outpath}")


class DoubleMuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            "muon_pt":  Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, 0, 100, name="val", label=r"Muon Leg $p_T$ [GeV]")),
            "muon_eta": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -2.5, 2.5, name="val", label=r"Muon Leg $\eta$")),
            "muon_phi": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"Muon Leg $\phi$")),
            "muon_dxy": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -100, 100, name="val", label=r"Muon Leg $d_{xy}$")),
            "muon_dz_overlay": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, -300, 300, name="val", label=r"Muon Leg $d_z$ [cm]")),
            "muon_validDTHits":   Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid DT Hits")),
            "muon_validCSCHits":  Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid CSC Hits")),
            "muon_validHits":     Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(80, 0, 80, name="val", label="Total Valid Muon Hits")),
            "muon_dtStations":    Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="DT Stations with Valid Hits")),
            "muon_timeAtIpInOut":    Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(25, -60, 60, name="val", label="Time at IP InOut [ns]")),
            "muon_timeAtIpInOutErr": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(75, 0, 5, name="val", label="Time at IP InOut Error [ns]")),
            "muon_timeNDof":         Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="timeNDof")),
            "muon_timing_err_ndof7":    Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (nDof > 7)")),
            "muon_timeErr_DT_only":     Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC == 0)")),
            "muon_timeErr_CSC_only":    Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT == 0, CSC > 0)")),
            "muon_timeErr_DT_CSC_both": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC > 0)")),
            "muon_timeErr_vs_DTHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Valid DT Hits"),
                axis.Regular(45, 0, 3,  name="err",  label="Time Error [ns]"),
            ),
            "muon_timeErr_vs_CSCHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Valid CSC Hits"),
                axis.Regular(45, 0, 3,  name="err",  label="Time Error [ns]"),
            ),
            "muon_timeErr_vs_TotalHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Total DT + CSC Hits"),
                axis.Regular(45, 0, 3,  name="err",  label="Time Error [ns]"),
            ),
            # ── Single muon histograms (exactly 1 DisMuon) ──────────────
            "single_muon_pt":  Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, 0, 100, name="val", label=r"Muon $p_T$ [GeV]")),
            "single_muon_eta": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -2.5, 2.5, name="val", label=r"Muon $\eta$")),
            "single_muon_phi": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"Muon $\phi$")),
            "single_muon_dxy": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -100, 100, name="val", label=r"Muon $d_{xy}$")),
            "single_muon_dz_overlay": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, -300, 300, name="val", label=r"Muon $d_z$ [cm]")),
            "single_muon_validDTHits":   Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid DT Hits")),
            "single_muon_validCSCHits":  Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid CSC Hits")),
            "single_muon_validHits":     Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(80, 0, 80, name="val", label="Total Valid Muon Hits")),
            "single_muon_dtStations":    Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="DT Stations with Valid Hits")),
            "single_muon_timeAtIpInOut":    Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(25, -60, 60, name="val", label="Time at IP InOut [ns]")),
            "single_muon_timeAtIpInOutErr": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(75, 0, 5, name="val", label="Time at IP InOut Error [ns]")),
            "single_muon_timeNDof":         Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="timeNDof")),
            "single_muon_timing_err_ndof7":    Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (nDof > 7)")),
            "single_muon_timeErr_DT_only":     Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC == 0)")),
            "single_muon_timeErr_CSC_only":    Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT == 0, CSC > 0)")),
            "single_muon_timeErr_DT_CSC_both": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC > 0)")),
            "single_muon_timeErr_vs_DTHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Valid DT Hits"),
                axis.Regular(45, 0, 3,  name="err",  label="Time Error [ns]"),
            ),
            "single_muon_timeErr_vs_CSCHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Valid CSC Hits"),
                axis.Regular(45, 0, 3,  name="err",  label="Time Error [ns]"),
            ),
            "single_muon_timeErr_vs_TotalHits": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(60, 0, 60, name="hits", label="Total DT + CSC Hits"),
                axis.Regular(45, 0, 3,  name="err",  label="Time Error [ns]"),
            ),
        }   # <-- plain closing brace, no stray )

    def fill_muon_hists(self, muons, cat_name, hist_prefix="muon"):
        # <-- body is now properly indented inside the method
        if len(muons) == 0:
            return

        self.output[f"{hist_prefix}_dz_overlay"].fill(cat=cat_name, val=muons.dz)
        self.output[f"{hist_prefix}_pt"].fill(cat=cat_name, val=muons.pt)
        self.output[f"{hist_prefix}_eta"].fill(cat=cat_name, val=muons.eta)
        self.output[f"{hist_prefix}_phi"].fill(cat=cat_name, val=muons.phi)
        self.output[f"{hist_prefix}_dxy"].fill(cat=cat_name, val=muons.dxy)
        self.output[f"{hist_prefix}_validDTHits"].fill(cat=cat_name, val=muons.numberOfValidMuonDTHits)
        self.output[f"{hist_prefix}_validCSCHits"].fill(cat=cat_name, val=muons.numberOfValidMuonCSCHits)
        self.output[f"{hist_prefix}_validHits"].fill(cat=cat_name, val=muons.numberOfValidMuonHits)
        self.output[f"{hist_prefix}_dtStations"].fill(cat=cat_name, val=muons.dtStationsWithValidHits)
        self.output[f"{hist_prefix}_timeAtIpInOut"].fill(cat=cat_name, val=muons.timeAtIpInOut)
        self.output[f"{hist_prefix}_timeAtIpInOutErr"].fill(cat=cat_name, val=muons.timeAtIpInOutErr)
        self.output[f"{hist_prefix}_timeNDof"].fill(cat=cat_name, val=muons.timeNDof)
        self.output[f"{hist_prefix}_timing_err_ndof7"].fill(cat=cat_name, val=muons[muons.timeNDof > 7].timeAtIpInOutErr)

        dt_only  = (muons.numberOfValidMuonDTHits > 0) & (muons.numberOfValidMuonCSCHits == 0)
        csc_only = (muons.numberOfValidMuonDTHits == 0) & (muons.numberOfValidMuonCSCHits > 0)
        both     = (muons.numberOfValidMuonDTHits > 0) & (muons.numberOfValidMuonCSCHits > 0)

        self.output[f"{hist_prefix}_timeErr_DT_only"].fill(cat=cat_name, val=muons[dt_only].timeAtIpInOutErr)
        self.output[f"{hist_prefix}_timeErr_CSC_only"].fill(cat=cat_name, val=muons[csc_only].timeAtIpInOutErr)
        self.output[f"{hist_prefix}_timeErr_DT_CSC_both"].fill(cat=cat_name, val=muons[both].timeAtIpInOutErr)

        self.output[f"{hist_prefix}_timeErr_vs_DTHits"].fill(cat=cat_name, hits=muons.numberOfValidMuonDTHits[dt_only], err=muons.timeAtIpInOutErr[dt_only])
        self.output[f"{hist_prefix}_timeErr_vs_CSCHits"].fill(cat=cat_name, hits=muons.numberOfValidMuonCSCHits[csc_only], err=muons.timeAtIpInOutErr[csc_only])
        self.output[f"{hist_prefix}_timeErr_vs_TotalHits"].fill(cat=cat_name, hits=(muons.numberOfValidMuonDTHits[both] + muons.numberOfValidMuonCSCHits[both]), err=muons.timeAtIpInOutErr[both])

    def process(self, events):
        dataset   = events.metadata.get("dataset", "Unknown")
        is_cosmic = "Cosmic" in dataset or dataset.startswith("LooseMu")
        prefix    = "MC" if is_cosmic else "Data"

        if not is_cosmic:
            events = events[events.HLT.L2Mu40_NoVertex_3Sta_NoBPTX3BX]
        #events = events[events.HLT.L2Mu40_NoVertex_3Sta_NoBPTX3BX]

        dismuon_mask = (events.DisMuon.pt > 40) & (abs(events.DisMuon.eta) < 2.4)
        events["DisMuon"] = events.DisMuon[dismuon_mask]

        n_dismuons = ak.num(events.DisMuon)

        # ── Exactly 1 DisMuon ────────────────────────────────────────
        mask_single = n_dismuons == 1
        if ak.sum(mask_single) > 0:
            single_events = events[mask_single]
            single_muons  = single_events.DisMuon[:, 0]

            quality_mask = (single_muons.mediumId == True) & (single_muons.pfRelIso03_all < 0.18)
            single_muons = single_muons[quality_mask]

            upper_singles = single_muons[single_muons.phi > 0]
            lower_singles = single_muons[single_muons.phi < 0]

            self.fill_muon_hists(upper_singles, f"Upper {prefix}", hist_prefix="single_muon")
            self.fill_muon_hists(lower_singles, f"Lower {prefix}", hist_prefix="single_muon")

        # ── Exactly 2 DisMuons ───────────────────────────────────────
        mask_exactly_two = n_dismuons == 2
        events = events[mask_exactly_two]

        if len(events) == 0:
            return self.output

        sorted_muons = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        lead_muon    = sorted_muons[:, 0]

        lead_quality_mask = (lead_muon.mediumId == True) & (lead_muon.pfRelIso03_all < 0.18)
        events       = events[lead_quality_mask]
        sorted_muons = sorted_muons[lead_quality_mask]

        if len(events) == 0:
            return self.output

        upper_candidates = sorted_muons[sorted_muons.phi > 0]
        lower_candidates = sorted_muons[sorted_muons.phi < 0]

        mask_one_up_one_down = (ak.num(upper_candidates) == 1) & (ak.num(lower_candidates) == 1)

        upper_muons = upper_candidates[mask_one_up_one_down][:, 0]
        lower_muons = lower_candidates[mask_one_up_one_down][:, 0]

        self.fill_muon_hists(upper_muons, f"Upper {prefix}")
        self.fill_muon_hists(lower_muons, f"Lower {prefix}")

        return self.output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":

    cosmic_pkl = "samples/Summer22_CHS_v19_Cosmic/Cosmic_preprocessed.pkl"
    print(f"Loading preprocessed Cosmics from {cosmic_pkl}...")
    with open(cosmic_pkl, "rb") as f:
        combined_runnable = pickle.load(f)

    nobptx_pkl = "samples/Summer22_CHS_v19_Cosmic/NoBPTX_preprocessed.pkl"
    print(f"Loading preprocessed NoBPTX from {nobptx_pkl}...")
    with open(nobptx_pkl, "rb") as f:
        nobptx_runnable = pickle.load(f)

    combined_runnable.update(nobptx_runnable)

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
        processor_instance=DoubleMuonProcessor(),
    )

    OUTPUT_DIR = "muon_leg_cosmic_vs_nobptx_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PREFIX         = "exact2mu_"
    TITLE_MODIFIER = "(Exact 2 DisMuon: Cosmic vs NoBPTX)"
    FILE_MODIFIER  = "Data_vs_MC"

    overlay_plots = [
        "muon_dz_overlay",
        "muon_validDTHits",
        "muon_validCSCHits",
        "muon_validHits",
        "muon_dtStations",
        "muon_pt",
        "muon_eta",
        "muon_phi",
        "muon_dxy",
        "muon_timeNDof",
        "muon_timeAtIpInOut",
        "muon_timeAtIpInOutErr",
        "muon_timing_err_ndof7",
        "muon_timeErr_DT_only",
        "muon_timeErr_CSC_only",
        "muon_timeErr_DT_CSC_both",
    ]

    profile_plots = [
        #"muon_timeErr_vs_DTHits",
        #"muon_timeErr_vs_CSCHits",
        #"muon_timeErr_vs_TotalHits",
    ]

    single_overlay_plots = [
        "single_muon_dz_overlay",
        "single_muon_validDTHits",
        "single_muon_validCSCHits",
        "single_muon_validHits",
        "single_muon_dtStations",
        "single_muon_pt",
        "single_muon_eta",
        "single_muon_phi",
        "single_muon_dxy",
        "single_muon_timeNDof",
        "single_muon_timeAtIpInOut",
        "single_muon_timeAtIpInOutErr",
        "single_muon_timing_err_ndof7",
        "single_muon_timeErr_DT_only",
        "single_muon_timeErr_CSC_only",
        "single_muon_timeErr_DT_CSC_both",
    ]

    single_profile_plots = [
        #"single_muon_timeErr_vs_DTHits",
        #"single_muon_timeErr_vs_CSCHits",
        #"single_muon_timeErr_vs_TotalHits",
    ]

    SINGLE_OUTPUT_DIR = "muon_leg_cosmic_vs_nobptx_plots/single_muon"
    os.makedirs(SINGLE_OUTPUT_DIR, exist_ok=True)

    for key, hist_obj in out.items():
        if isinstance(hist_obj, (int, float)):
            continue

        if key in overlay_plots:
            save_comparison_overlay(hist_obj, key, PREFIX, OUTPUT_DIR,
                title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER, normalize=True)

        elif key in profile_plots:
            save_2d_profile_overlay(hist_obj, key, PREFIX, OUTPUT_DIR,
                title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER)

        elif key in single_overlay_plots:
            save_comparison_overlay(hist_obj, key, "single_", SINGLE_OUTPUT_DIR,
                title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER, normalize=True,
                ratio_ylim=(0, 2.5))

        elif key in single_profile_plots:
            save_2d_profile_overlay(hist_obj, key, "single_", SINGLE_OUTPUT_DIR,
                title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER)

    print("Done!")