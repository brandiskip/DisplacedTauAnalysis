import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import mplhep as hep

hep.style.use("CMS")

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

LABEL_A = "CollisionCalib"
LABEL_B = "CosmicCalib"


def save_comparison_overlay(
    h, var_name, PREFIX, OUTPUT_DIR,
    log_y=False, normalize=True, ratio_ylim=(0.5, 1.5),
):
    if np.sum(h.values()) == 0:
        print(f"  Skipping {var_name} (empty)")
        return

    all_cats = list(h.axes["cat"])

    if LABEL_A not in all_cats or LABEL_B not in all_cats:
        print(f"  Skipping {var_name}: need {LABEL_A!r} and {LABEL_B!r}, have {all_cats}")
        return

    h_a = h[{"cat": LABEL_A}]
    h_b = h[{"cat": LABEL_B}]

    raw_a = h_a.values()
    raw_b = h_b.values()
    total_a = np.sum(raw_a)
    total_b = np.sum(raw_b)

    if total_a == 0 or total_b == 0:
        return

    if normalize:
        h_a_plot = h_a * (1.0 / total_a)
        h_b_plot = h_b * (1.0 / total_b)
    else:
        h_a_plot = h_a
        h_b_plot = h_b

    vals_a = h_a_plot.values()
    vals_b = h_b_plot.values()
    bin_centers = h_a_plot.axes[0].centers
    bin_edges   = h_a_plot.axes[0].edges
    bin_width   = bin_edges[1] - bin_edges[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vals_a > 0, vals_b / vals_a, np.nan)
        ratio_err = np.where(
            raw_b > 0,
            ratio * np.sqrt(1.0 / raw_b),
            np.nan,
        )

    fig = plt.figure(figsize=(8, 9))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    ax_main.stairs(
        vals_a, bin_edges,
        color="#5790fc", linewidth=1.5,
        fill=True, alpha=0.4, label=f"{LABEL_A}  (N={int(total_a)})",
    )
    ax_main.stairs(vals_a, bin_edges, color="#5790fc", linewidth=1.5)

    ax_main.errorbar(
        bin_centers, vals_b,
        yerr=np.where(raw_b > 0, vals_b / np.sqrt(raw_b), 0),
        fmt="+", color="#e42536", markersize=6, elinewidth=1,
        capsize=2, label=f"{LABEL_B}  (N={int(total_b)})",
    )

    ylabel = (
        f"Fraction of Events / {bin_width:.2g}" if normalize
        else f"Events / {bin_width:.2g}"
    )
    ax_main.set_ylabel(ylabel)
    hep.cms.label(data=False, label="Private Work", com=13.6, ax=ax_main, fontsize=13)
    ax_main.set_title(var_name, fontsize=11, loc="center", pad=25)
    ax_main.legend(frameon=False, fontsize=9)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    ax_main.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="lower"))

    if log_y:
        ax_main.set_yscale("log")

    ax_ratio.errorbar(
        bin_centers, ratio, yerr=ratio_err,
        fmt="o", color="black", markersize=3, elinewidth=1, capsize=2,
    )
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_ratio.set_ylabel(f"{LABEL_B} / {LABEL_A}", fontsize=10)
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.set_xlabel(h_a_plot.axes[0].label)
    ax_ratio.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax_ratio.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax_ratio.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {outpath}")


# ── Histogram definitions (no timing variables) ─────────────────────────────

def make_hists():
    cat = axis.StrCategory([], name="cat", label="Sample", growth=True)
    return {
        "muon_pt":           Hist(cat, axis.Regular(100, 0,    300, name="val", label=r"DisMuon $p_T$ [GeV]")),
        "muon_eta":          Hist(cat, axis.Regular(100, -2.5, 2.5, name="val", label=r"DisMuon $\eta$")),
        "muon_phi":          Hist(cat, axis.Regular(100, -np.pi, np.pi, name="val", label=r"DisMuon $\phi$")),
        "muon_dxy":          Hist(cat, axis.Regular(100, -100, 100, name="val", label=r"DisMuon $d_{xy}$ [cm]")),
        "muon_dz":           Hist(cat, axis.Regular(160, -800, 800, name="val", label=r"DisMuon $d_z$ [cm]")),
        "muon_validDTHits":  Hist(cat, axis.Regular(60,  0,    60,  name="val", label="Valid DT Hits")),
        "muon_validCSCHits": Hist(cat, axis.Regular(60,  0,    60,  name="val", label="Valid CSC Hits")),
        "muon_validHits":    Hist(cat, axis.Regular(80,  0,    80,  name="val", label="Total Valid Muon Hits")),
        "muon_dtStations":   Hist(cat, axis.Regular(10,  0,    10,  name="val", label="DT Stations with Valid Hits")),
        "muon_nDisMuons":    Hist(cat, axis.Regular(10,  0,    10,  name="val", label="Number of DisMuons per Event")),
        "n_events_before_cuts": Hist(cat, axis.Regular(1, -0.5, 0.5, name="val", label="Event Count")),
    }


class CalibComparisonProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = make_hists()

    def fill_muon_hists(self, muons, cat_name):
        if len(muons) == 0:
            return

        self.output["muon_pt"].fill(cat=cat_name, val=muons.pt)
        self.output["muon_eta"].fill(cat=cat_name, val=muons.eta)
        self.output["muon_phi"].fill(cat=cat_name, val=muons.phi)
        self.output["muon_dxy"].fill(cat=cat_name, val=muons.dxy)
        self.output["muon_dz"].fill(cat=cat_name, val=muons.dz)
        self.output["muon_validDTHits"].fill(cat=cat_name, val=muons.numberOfValidMuonDTHits)
        self.output["muon_validCSCHits"].fill(cat=cat_name, val=muons.numberOfValidMuonCSCHits)
        self.output["muon_validHits"].fill(cat=cat_name, val=muons.numberOfValidMuonHits)
        self.output["muon_dtStations"].fill(cat=cat_name, val=muons.dtStationsWithValidHits)

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")

        if LABEL_A in dataset:
            sample_label = LABEL_A
        elif LABEL_B in dataset:
            sample_label = LABEL_B
        else:
            print(f"  WARNING: dataset {dataset!r} does not contain {LABEL_A!r} or {LABEL_B!r}")
            sample_label = dataset

        self.output["n_events_before_cuts"].fill(cat=sample_label, val=np.zeros(len(events)))

        # Kinematic selection only
        #dismuon_mask = (events.DisMuon.pt > 20) & (abs(events.DisMuon.eta) < 2.4)
        #events["DisMuon"] = events.DisMuon[dismuon_mask]

        # Fill multiplicity before flattening
        self.output["muon_nDisMuons"].fill(cat=sample_label, val=ak.num(events.DisMuon))

        # Flatten all DisMuons
        all_muons = ak.flatten(events.DisMuon)

        if len(all_muons) == 0:
            return self.output

        self.fill_muon_hists(all_muons, sample_label)

        return self.output

    def postprocess(self, accumulator):
        return accumulator


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # # ── Original pickle-based input (full samples) ──────────────────────
    # pkl_a = "samples/Summer22_CHS_collisionCalib_v19_Cosmic/Cosmic_CollisionCalib_preprocessed.pkl"
    # pkl_b = "samples/Summer22_CHS_v17_Cosmic/Cosmic_CosmicCalib_preprocessed.pkl"
    #
    # print(f"Loading {LABEL_A} from {pkl_a} ...")
    # with open(pkl_a, "rb") as f:
    #     runnable_a = pickle.load(f)
    #
    # print(f"Loading {LABEL_B} from {pkl_b} ...")
    # with open(pkl_b, "rb") as f:
    #     runnable_b = pickle.load(f)
    #
    # print(f"\nDataset keys in {LABEL_A} pkl: {list(runnable_a.keys())}")
    # print(f"Dataset keys in {LABEL_B} pkl: {list(runnable_b.keys())}")
    #
    # combined = {}
    # for key, val in runnable_a.items():
    #     combined[f"{LABEL_A}__{key}"] = val
    # for key, val in runnable_b.items():
    #     combined[f"{LABEL_B}__{key}"] = val
    #
    # print(f"\nCombined dict has {len(combined)} dataset keys")
    # print(f"Keys: {list(combined.keys())}")

    # ── Small-sample input (Sara's files) ─────────────────────────────────
    base_dir = "/afs/cern.ch/user/f/fiorendi/public/displacedTaus/forBrandi/nano_cosmic"

    combined = {
        f"{LABEL_A}__nano": [f"{base_dir}/nano_collisionCalibration.root"],
        f"{LABEL_B}__nano": [f"{base_dir}/nano_cosmicCalibration.root"],
    }

    print(f"Combined dict has {len(combined)} dataset keys")
    print(f"Keys: {list(combined.keys())}")

    print("\nRunning processor ...")
    executor = processor.FuturesExecutor(workers=8)
    runner   = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
        skipbadfiles=True,
    )

    out = runner(
        combined,
        treename="Events",
        processor_instance=CalibComparisonProcessor(),
    )

    sample_hist = out.get("muon_pt")
    if sample_hist is not None:
        print(f"\nCategories in muon_pt histogram: {list(sample_hist.axes['cat'])}")
        print(f"Total entries in muon_pt: {np.sum(sample_hist.values())}")

    # ── Output ────────────────────────────────────────────────────────────
    OUTPUT_DIR = "sara_calib_comparison_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nEvent counts before kinematic cuts:")
    for label in [LABEL_A, LABEL_B]:
        h = out["n_events_before_cuts"][{"cat": label}]
        print(f"  {label}: {int(np.sum(h.values())):,} events")

    overlay_vars = [
        "muon_pt",
        "muon_eta",
        "muon_phi",
        "muon_dxy",
        "muon_dz",
        "muon_validDTHits",
        "muon_validCSCHits",
        "muon_validHits",
        "muon_dtStations",
        "muon_nDisMuons",
    ]

    print("\nSaving plots ...")
    for var in overlay_vars:
        if var not in out:
            print(f"  {var} not found in output, skipping")
            continue
        save_comparison_overlay(
            out[var], var, "all_", OUTPUT_DIR,
            normalize=True, ratio_ylim=(0.5, 1.5),
        )

    print("\nDone!")