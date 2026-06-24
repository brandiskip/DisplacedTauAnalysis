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

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

NDOF_MIN = 7   # timeNDof must be > this on both hemisphere-leading muons before Delta t is trusted


def _as_bool(x):
    """Cast a NanoAOD flag (bool or int8 0/1) to a real boolean awkward array so
    that logical ~ / & behave correctly."""
    return ak.values_astype(x, np.bool_)


def get_Lxy(genvistau):
    """Transverse decay length: tau production vertex (= stau decay vertex)
    minus the stau production vertex (= PV)."""
    vx = genvistau.parent.vx - genvistau.parent.distinctParent.vx
    vy = genvistau.parent.vy - genvistau.parent.distinctParent.vy
    return np.sqrt(vx ** 2 + vy ** 2)


def plot_cosA_efficiency(h_all, h_post, OUTPUT_DIR, PREFIX, title_suffix, filename_suffix, title):
    if np.sum(h_all.values()) == 0:
        print("Skipping efficiency plot (empty denominator)")
        return

    fig, (ax_dist, ax_eff) = plt.subplots(
        2, 1, figsize=(8, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    all_vals  = h_all.values()
    post_vals = h_post.values()
    centers   = h_all.axes["val"].centers
    edges     = h_all.axes["val"].edges

    ax_dist.step(edges[:-1], all_vals,  where="post", label="All Signal",  color="blue", linestyle="--")
    ax_dist.step(edges[:-1], post_vals, where="post", label="Post-veto",   color="red",  linestyle="-")
    ax_dist.set_ylabel("Events")
    ax_dist.set_title(title + " " + title_suffix)
    ax_dist.legend()

    eff = np.where(all_vals > 0, post_vals / all_vals, np.nan)
    err = np.where(all_vals > 0,
                   np.sqrt(eff * (1 - np.clip(eff, 0, 1)) / np.clip(all_vals, 1, None)),
                   np.nan)
    ax_eff.errorbar(centers, eff, yerr=err, fmt="o-", color="black", capsize=3, markersize=4)
    ax_eff.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax_eff.set_xlabel(r"Leading DisMuon $p_T$ [GeV]")
    ax_eff.set_ylabel("Efficiency")
    ax_eff.set_ylim(-0.05, 1.15)

    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}efficiency_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved efficiency plot: {outpath}")


def plot_ndof_lead_sub(h_lead, h_sub, OUTPUT_DIR, PREFIX, title_suffix, filename_suffix):
    """timeNDof of the two muons used for the Delta t computation, separated by
    whether the muon is the leading-pT or subleading-pT member of the pair.
    The vertical line at NDOF_MIN shows where the ndof cut bites."""
    if (np.sum(h_lead.values()) + np.sum(h_sub.values())) == 0:
        print("Skipping ndof(lead/sub) plot (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = h_lead.axes["val"].edges
    ax.step(edges[:-1], h_lead.values(), where="post", color="blue", label=r"Leading-$p_T$ $\Delta t$ muon")
    ax.step(edges[:-1], h_sub.values(),  where="post", color="red",  label=r"Subleading-$p_T$ $\Delta t$ muon")
    ax.axvline(NDOF_MIN, color="green", linestyle="--", label=f"ndof > {NDOF_MIN} cut")
    ax.set_xlabel("DisMuon timeNDof")
    ax.set_ylabel("DisMuons (one entry per hemisphere-leader)")
    ax.set_title("timeNDof of the two Delta-t muons " + title_suffix)
    ax.legend()
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}ndof_leadsub_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved ndof lead/sub plot: {outpath}")


def plot_dt(h, OUTPUT_DIR, PREFIX, title_suffix, filename_suffix):
    """Delta t for events that survived cosA AND passed ndof>7 on both hemisphere
    leaders (i.e. the events the Delta t < -20 ns veto actually acts on)."""
    if np.sum(h.values()) == 0:
        print("Skipping Delta t plot (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = h.axes["val"].edges
    ax.step(edges[:-1], h.values(), where="post", color="purple")
    ax.axvline(-20, color="red", linestyle="--", label=r"$\Delta t < -20$ ns veto")
    ax.set_xlabel(r"$\Delta t = t_{\mathrm{upper}} - t_{\mathrm{lower}}$ [ns]")
    ax.set_ylabel("Events (ndof > 7 on both leaders)")
    ax.set_title(r"$\Delta t$ of ndof-passing events " + title_suffix)
    ax.legend()
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}deltat_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved Delta t plot: {outpath}")


def plot_dtveto_muon_types(lead_counts, sub_counts, OUTPUT_DIR, PREFIX, title_suffix, filename_suffix):
    """Mutually-exclusive muon type of the two Delta-t muons, ONLY for events the
    Delta t < -20 ns cut vetoes.  Each event contributes exactly one leading and one
    subleading muon.  'Standalone only' is the bucket that fails Loose ID
    (Loose ID = isGlobal || isTracker)."""
    labels = ["Global", "Tracker\n(not global)", "Standalone\nonly", "None"]
    if sum(lead_counts) + sum(sub_counts) == 0:
        print("Skipping dt-veto muon-type plot (no dt-vetoed events)")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2, lead_counts, w, label=r"Leading-$p_T$ $\Delta t$ muon",    color="#4C72B0")
    b2 = ax.bar(x + w / 2, sub_counts,  w, label=r"Subleading-$p_T$ $\Delta t$ muon", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"DisMuons (one per $\Delta t$-vetoed event)")
    ax.set_xlabel("Muon type (mutually exclusive)")
    ax.set_title(r"Muon type in $\Delta t<-20$ ns vetoed events " + title_suffix)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")
    # shade the "Standalone only" column -> the muons a Loose ID cut would remove
    ax.axvspan(x[2] - 0.5, x[2] + 0.5, color="orange", alpha=0.12)
    ax.text(x[2], ax.get_ylim()[1] * 0.92, "fails Loose ID\n(isGlobal||isTracker)",
            ha="center", va="top", fontsize=8, color="darkorange")
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}dtveto_muon_types_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved dt-veto muon-type plot: {outpath}")


def plot_time2d(h, title, OUTPUT_DIR, PREFIX, filename_suffix):
    """2D timeAtIpInOut: leading-pT muon (x) vs subleading-pT muon (y) of the
    Delta-t pair.  In-time leading muons should cluster at x~0; out-of-time
    subleading muons spread to y = +-25, +-50 ns (the 25 ns LHC bunch structure)."""
    if np.sum(h.values()) == 0:
        print(f"Skipping 2D time plot {filename_suffix} (empty)")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    h.plot2d(ax=ax, norm=mcolors.LogNorm(vmin=1))
    for t in (-50, -25, 0, 25, 50):
        ax.axhline(t, color="white", lw=0.6, ls=":", alpha=0.5)
        ax.axvline(t, color="white", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel(r"Leading-$p_T$ muon  timeAtIpInOut [ns]")
    ax.set_ylabel(r"Subleading-$p_T$ muon  timeAtIpInOut [ns]")
    ax.set_title(title)
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}time2d_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved 2D time plot: {outpath}")


def plot_sub_time(h, OUTPUT_DIR, PREFIX, filename_suffix, title):
    """1D timeAtIpInOut of the subleading dt muon (ndof>7).  Vertical lines mark
    the 25 ns bunch structure and the |t| = 12.5 ns (half-BX) in/out-of-time split."""
    if np.sum(h.values()) == 0:
        print(f"Skipping {filename_suffix} (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = h.axes["val"].edges
    ax.step(edges[:-1], h.values(), where="post", color="purple")
    for t in (-50, -25, 25, 50):
        ax.axvline(t, color="gray", ls=":", alpha=0.6)
    ax.axvline(12.5,  color="red", ls="--", alpha=0.8, label="|t| = 12.5 ns")
    ax.axvline(-12.5, color="red", ls="--", alpha=0.8)
    ax.set_xlabel(h.axes["val"].label)
    ax.set_ylabel("Subleading muons / bin")
    ax.set_title(title)
    ax.legend()
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved subleading-time plot: {outpath}")


def plot_sub_overlay(h, OUTPUT_DIR, PREFIX, filename_suffix, title):
    """Overlay a subleading-muon property for in-time vs out-of-time |t|,
    area-normalized so the shapes compare regardless of yield."""
    if np.sum(h.values()) == 0:
        print(f"Skipping overlay {filename_suffix} (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = h.axes["val"].edges
    for cat in h.axes["tcat"]:
        vals = h[{"tcat": cat}].values()
        tot  = vals.sum()
        if tot == 0:
            continue
        ax.step(edges[:-1], vals / tot, where="post", label=f"{cat}  (N={int(tot)})")
    ax.set_xlabel(h.axes["val"].label)
    ax.set_ylabel("Fraction of pairs / bin")
    ax.set_title(title)
    ax.legend()
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved overlay plot: {outpath}")


def plot_2d_by_time(h, OUTPUT_DIR, PREFIX, base_suffix, title):
    """2D leading (x) vs subleading (y) muon variable, one PDF per in/out-of-time
    category (split by the subleading muon's timing)."""
    for cat in h.axes["tcat"]:
        hc = h[{"tcat": cat}]
        if np.sum(hc.values()) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        hc.plot2d(ax=ax, norm=mcolors.LogNorm(vmin=1))
        ax.set_title(f"{title}  [{cat}]")
        safe = cat.replace("-", "")
        outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{base_suffix}_{safe}.pdf")
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved 2D plot: {outpath}")


class CosASignalCheckProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            "n_events_initial":            0,
            "n_gen_signal":                0,   # gen mu+tau region: 1 GenVisStauTau + 1 GenMuon + 0 GenElectron
            "n_after_jetmet_mu":           0,   # jet==1, MET>105, >=1 DisMuon (before leading quality)
            "n_signal_denom":              0,   # after leading-muon quality cuts  (veto denominator)
            "n_events_2plus_muons":        0,
            "n_events_2plus_muons_postdup":0,
            "n_duplicate_tracks_removed":  0,

            "n_events_vetoed_cosA":        0,   # cosA < -0.99 veto

            # --- timing veto, correctly gated on ndof>7 -----------------------
            "n_events_pair_present":       0,   # post-cosA, an upper AND a lower hemisphere muon exist
            "n_events_excluded_ndof":      0,   # pair present but ndof<=7 on a leader -> NOT eligible for dt veto
            "n_events_vetoed_ndof_dt":     0,   # ndof>7 on both AND dt < -20 ns  (the real timing veto)
            "n_dt_veto_no_ndof":           0,   # OLD/buggy: dt < -20 ignoring ndof  (for comparison)

            # ndof on the two Delta-t muons, split leading vs subleading
            "n_leadmu_ndof_pass":          0,
            "n_leadmu_ndof_fail":          0,
            "n_submu_ndof_pass":           0,
            "n_submu_ndof_fail":           0,

            "n_numer_cosA":                0,   # survive cosA
            "n_numer_full":                0,   # survive cosA + (ndof+dt)

            # dt (upper-lower) BX structure for ndof>7 pairs: centered at 0 vs 25 ns satellites
            "n_dt_ndofok":   0,   # total ndof>7 pairs (denominator)
            "n_dt_central":  0,   # |dt| < 12.5 ns                 (centered at zero)
            "n_dt_bx1":      0,   # 12.5 <= |dt| < 37.5 ns         (~ +-25 ns, 1 BX off)
            "n_dt_bx2":      0,   # 37.5 <= |dt| < 62.5 ns         (~ +-50 ns, 2 BX off)
            "n_dt_bxhi":     0,   # |dt| >= 62.5 ns                (>=3 BX off)

            # mutually-exclusive muon type of the two dt muons, ONLY for dt-vetoed events
            "n_dtveto_lead_global":        0,
            "n_dtveto_lead_tracker":       0,   # tracker but not global
            "n_dtveto_lead_saonly":        0,   # standalone, not global, not tracker  (fails Loose ID)
            "n_dtveto_lead_none":          0,
            "n_dtveto_sub_global":         0,
            "n_dtveto_sub_tracker":        0,
            "n_dtveto_sub_saonly":         0,
            "n_dtveto_sub_none":           0,

            "lead_pt_all":       Hist(axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")),
            "lead_pt_post_cosA": Hist(axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")),
            "lead_pt_post_full": Hist(axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")),

            # timeNDof of the two Delta-t muons (all pair-present events), split lead/sub
            "ndof_lead": Hist(axis.Regular(50, 0, 50, name="val", label="timeNDof")),
            "ndof_sub":  Hist(axis.Regular(50, 0, 50, name="val", label="timeNDof")),
            # Delta t for events that pass ndof>7 on both leaders
            "dt_ndofpass": Hist(axis.Regular(100, -100, 100, name="val", label=r"$\Delta t$ [ns]")),
            # timeNDof of all selected muons (kept from your original script)
            "timeNDof_all": Hist(axis.Regular(50, 0, 50, name="val", label="DisMuon timeNDof")),

            # ── 2D timeAtIpInOut: leading-pT (x) vs subleading-pT (y) muon of the
            #    Delta-t pair, split by the subleading muon's (mutually exclusive) type ──
            "time_lead_vs_sub_global": Hist(
                axis.Regular(80, -80, 80, name="lead", label=r"Leading $\mu$ timeAtIpInOut [ns]"),
                axis.Regular(80, -80, 80, name="sub",  label=r"Subleading $\mu$ timeAtIpInOut [ns]"),
            ),
            "time_lead_vs_sub_tracker": Hist(
                axis.Regular(80, -80, 80, name="lead", label=r"Leading $\mu$ timeAtIpInOut [ns]"),
                axis.Regular(80, -80, 80, name="sub",  label=r"Subleading $\mu$ timeAtIpInOut [ns]"),
            ),
            "time_lead_vs_sub_standalone": Hist(
                axis.Regular(80, -80, 80, name="lead", label=r"Leading $\mu$ timeAtIpInOut [ns]"),
                axis.Regular(80, -80, 80, name="sub",  label=r"Subleading $\mu$ timeAtIpInOut [ns]"),
            ),

            # ── Investigative: subleading dt-muon time (shows the 25 ns comb) ──
            "sub_time": Hist(axis.Regular(120, -75, 75, name="val", label=r"Subleading $\mu$ timeAtIpInOut [ns]")),

            # ── Per-variable overlay: leading vs subleading(in-time) vs subleading(out-of-time).
            #    The "tcat" axis carries the 3 categories; leading is NOT split (it's in-time). ──
            "prop_pt":      Hist(axis.StrCategory([], name="tcat", growth=True), axis.Regular(50, 0, 500,   name="val", label=r"Muon $p_T$ [GeV]")),
            "prop_eta":     Hist(axis.StrCategory([], name="tcat", growth=True), axis.Regular(50, -2.5, 2.5, name="val", label=r"Muon $\eta$")),
            "prop_dxy":     Hist(axis.StrCategory([], name="tcat", growth=True), axis.Regular(60, -30, 30,   name="val", label=r"Muon $d_{xy}$ [cm]")),
            "prop_dz":      Hist(axis.StrCategory([], name="tcat", growth=True), axis.Regular(60, -100, 100, name="val", label=r"Muon $d_z$ [cm]")),
            "prop_dxybs":   Hist(axis.StrCategory([], name="tcat", growth=True), axis.Regular(60, -30, 30,   name="val", label=r"Muon $d_{xy}^{BS}$ [cm]")),
            "prop_timeErr": Hist(axis.StrCategory([], name="tcat", growth=True), axis.Regular(60, 0, 6,      name="val", label=r"Muon timeAtIpInOutErr [ns]")),

            # ── cosα between the leading and subleading dt muon ──
            "cosA_lead_sub_by_time": Hist(axis.StrCategory([], name="tcat", growth=True), axis.Regular(100, -1.0, 1.0, name="val", label=r"$\cos\alpha$(lead, sub)")),

            # ── 2D leading vs subleading: eta, pt, phi ──
            "eta2d_lead_sub": Hist(axis.StrCategory([], name="tcat", growth=True),
                                   axis.Regular(50, -2.5, 2.5, name="lead", label=r"Leading $\mu$ $\eta$"),
                                   axis.Regular(50, -2.5, 2.5, name="sub",  label=r"Subleading $\mu$ $\eta$")),
            "pt2d_lead_sub":  Hist(axis.StrCategory([], name="tcat", growth=True),
                                   axis.Regular(50, 0, 500, name="lead", label=r"Leading $\mu$ $p_T$ [GeV]"),
                                   axis.Regular(50, 0, 200, name="sub",  label=r"Subleading $\mu$ $p_T$ [GeV]")),
            "phi2d_lead_sub": Hist(axis.StrCategory([], name="tcat", growth=True),
                                   axis.Regular(50, -np.pi, np.pi, name="lead", label=r"Leading $\mu$ $\phi$"),
                                   axis.Regular(50, -np.pi, np.pi, name="sub",  label=r"Subleading $\mu$ $\phi$")),
        }

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")
        self.output["n_events_initial"] += len(events)

        is_signal = not ("Cosmic" in dataset or dataset.startswith("LooseMu")
                         or dataset == "test_cosmics_calib" or "NoBPTX" in dataset)
        has_gen = "GenPart" in events.fields

        if not (is_signal and has_gen):
            return self.output

        # ════════════════════════════════════════════════════════════
        # GEN-LEVEL SIGNAL REGION (applied BEFORE any DisMuon cuts):
        #   exactly 1 GenVisStauTau, exactly 1 GenMuon, 0 GenElectrons
        # ════════════════════════════════════════════════════════════
        gpart = events.GenPart
        events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & gpart.hasFlags("isLastCopy")]
        events['staus_taus'] = events.staus.distinctChildren[
            (abs(events.staus.distinctChildren.pdgId) == 15) &
            events.staus.distinctChildren.hasFlags("isLastCopy") &
            events.staus.distinctChildren.hasFlags("fromHardProcess")
        ]

        genvistau_Lxy = get_Lxy(events.GenVisTau)
        events['GenVisStauTaus'] = events.GenVisTau[
            (abs(events.GenVisTau.parent.pdgId) == 15) &
            (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
            events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy") &
            events.GenVisTau.parent.hasFlags("fromHardProcess") &
            (genvistau_Lxy < 100.0) &
            (events.GenVisTau.pt > 20) &
            (abs(events.GenVisTau.eta) < 2.4)
        ]
        d0 = abs(
            (events.GenVisStauTaus.parent.vy - events.GenVtx.y) * np.cos(events.GenVisStauTaus.parent.phi) -
            (events.GenVisStauTaus.parent.vx - events.GenVtx.x) * np.sin(events.GenVisStauTaus.parent.phi)
        )
        events['GenVisStauTaus'] = ak.with_field(events.GenVisStauTaus, d0, where="d0")

        events['GenMuon'] = gpart[(abs(gpart.pdgId) == 13) & gpart.hasFlags("isLastCopy")]
        events['GenMuon'] = events.GenMuon[
            (events.GenMuon.pt > 20) &
            (abs(events.GenMuon.eta) < 2.4) &
            (abs(events.GenMuon.distinctParent.distinctParent.pdgId) == 1000015)
        ]

        events['GenElectron'] = events.GenPart[(abs(events.GenPart.pdgId) == 11) & events.GenPart.hasFlags("isLastCopy")]
        events['GenElectron'] = events.GenElectron[
            (events.GenElectron.pt > 20) &
            (abs(events.GenElectron.eta) < 2.4) &
            (abs(events.GenElectron.distinctParent.distinctParent.pdgId) == 1000015)
        ]

        gen_mask = (
            (ak.num(events.GenVisStauTaus) == 1) &
            (ak.num(events.GenMuon) == 1) &
            (ak.num(events.GenElectron) == 0)
        )
        self.output["n_gen_signal"] += int(ak.sum(gen_mask))
        events = events[gen_mask]
        if len(events) == 0:
            return self.output

        events["DisMuon"] = ak.zip(
            {
                "pt":                       events.DisMuon.pt,
                "eta":                      events.DisMuon.eta,
                "phi":                      events.DisMuon.phi,
                "mass":                     events.DisMuon.mass,
                "charge":                   events.DisMuon.charge,
                "timeNDof":                 events.DisMuon.timeNDof,
                "dxy":                      events.DisMuon.dxy,
                "dxybs":                    events.DisMuon.dxybs,
                "dz":                       events.DisMuon.dz,
                "numberOfValidMuonDTHits":  events.DisMuon.numberOfValidMuonDTHits,
                "numberOfValidMuonCSCHits": events.DisMuon.numberOfValidMuonCSCHits,
                "numberOfValidMuonHits":    events.DisMuon.numberOfValidMuonHits,
                "dtStationsWithValidHits":  events.DisMuon.dtStationsWithValidHits,
                "timeAtIpInOut":            events.DisMuon.timeAtIpInOut,
                "timeAtIpInOutErr":         events.DisMuon.timeAtIpInOutErr,
                "mediumId":                 events.DisMuon.mediumId,
                "pfRelIso03_all":           events.DisMuon.pfRelIso03_all,
                "isStandalone":             events.DisMuon.isStandalone,
                "isGlobal":                 events.DisMuon.isGlobal,
                "isTracker":                events.DisMuon.isTracker,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # NEW selection: apply the pt/eta cut to the LEADING (highest-pT) DisMuon
        # only; sub-leading muons are kept with no pt/eta requirement.  Events whose
        # leading muon fails lose their whole DisMuon collection.
        sorted_all  = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        lead_all    = ak.firsts(sorted_all)
        lead_passes = ak.fill_none((lead_all.pt > 30) & (abs(lead_all.eta) < 2.4), False)

        keep_per_muon, _ = ak.broadcast_arrays(lead_passes, sorted_all.pt)
        events["DisMuon"] = sorted_all[keep_per_muon]

        # ── Signal-only event filtering (jets + MET) ──
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.where(
            ak.all(events.Jet.constituents.pf.charge == 0, axis=-1),
            -999,
            ak.flatten(
                events.Jet.constituents.pf[
                    ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)
                ].d0,
                axis=-1,
            ),
        )
        dxy = ak.fill_none(dxy, -999)
        events["Jet"] = ak.with_field(events.Jet, dxy, where="dxy")

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

        good_MET    = events.PFMET.pt > 105
        signal_mask = (ak.num(jets) == 1) & good_MET
        events      = events[signal_mask]

        # ── Require at least 1 DisMuon ──
        mask_has_muons = ak.num(events.DisMuon) > 0
        if ak.sum(mask_has_muons) == 0:
            return self.output
        events = events[mask_has_muons]
        self.output["n_after_jetmet_mu"] += len(events)

        # ── Sort by pT, apply quality cuts to the leading muon only ──
        sorted_muons = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        lead = sorted_muons[:, 0]

        lead_quality_mask = (
            (lead.mediumId == True) &
            (lead.pfRelIso03_all < 0.18) &
            (abs(lead.dxy) > 0.1) &
            (abs(lead.dxy) < 10)
        )

        sorted_muons = sorted_muons[lead_quality_mask]
        if len(sorted_muons) == 0:
            return self.output

        lead    = sorted_muons[:, 0]
        n_muons = ak.num(sorted_muons)

        # ══════════════════════════════════════════
        # DENOMINATOR: signal events after full selection (incl. leading quality)
        # ══════════════════════════════════════════
        self.output["n_signal_denom"] += len(sorted_muons)
        self.output["lead_pt_all"].fill(val=lead.pt)
        self.output["timeNDof_all"].fill(val=ak.to_numpy(ak.flatten(sorted_muons.timeNDof)))

        # ── 2+ muon count, and the same after duplicate-track removal ──
        self.output["n_events_2plus_muons"] += int(ak.sum(n_muons >= 2))

        local_i = ak.local_index(sorted_muons, axis=1)
        a, b    = ak.unzip(ak.cartesian([sorted_muons, sorted_muons], axis=1, nested=True))
        ia, ib  = ak.unzip(ak.cartesian([local_i, local_i], axis=1, nested=True))
        deta = a.eta - b.eta
        dphi = a.delta_phi(b)
        dpt  = a.pt - b.pt
        mask_sc = (a.charge * b.charge) > 0
        match = (
            mask_sc
            & (abs(deta) < 0.01)
            & (abs(dphi) < 0.001)
            & (abs(dpt)  < 0.5)
            & (ib < ia)
        )
        is_duplicate    = ak.any(match, axis=2)
        sorted_muons_dd = sorted_muons[~is_duplicate]
        self.output["n_events_2plus_muons_postdup"] += int(ak.sum(ak.num(sorted_muons_dd) >= 2))
        self.output["n_duplicate_tracks_removed"]   += int(
            ak.sum(ak.num(sorted_muons)) - ak.sum(ak.num(sorted_muons_dd))
        )

        # ══════════════════════════════════════════
        # cosA VETO: for events with 2+ DisMuons, compute cosA between the leading
        # muon and every subsequent muon.  If ANY pair has cosA < -0.99, veto event.
        # ══════════════════════════════════════════
        has_2plus = (n_muons >= 2)
        survives_cosA = np.ones(len(sorted_muons), dtype=bool)

        if ak.sum(has_2plus) > 0:
            multi_muons = sorted_muons[has_2plus]
            multi_lead  = multi_muons[:, 0]
            sub_muons   = multi_muons[:, 1:]

            dot = (multi_lead.px * sub_muons.px +
                   multi_lead.py * sub_muons.py +
                   multi_lead.pz * sub_muons.pz)
            mag = multi_lead.p * sub_muons.p
            cosA = ak.where(mag > 0, dot / mag, -1000.0)

            any_backtoback = ak.any(cosA < -0.99, axis=1)

            pair_idx = np.where(ak.to_numpy(has_2plus))[0]
            survives_cosA[pair_idx] = ~ak.to_numpy(any_backtoback)

        self.output["n_events_vetoed_cosA"] += int(np.sum(~survives_cosA))
        self.output["n_numer_cosA"]         += int(np.sum(survives_cosA))
        self.output["lead_pt_post_cosA"].fill(val=lead.pt[ak.Array(survives_cosA)])

        # ══════════════════════════════════════════════════════════════════
        # TIMING VETO  (correctly gated on ndof > 7)
        #   For events surviving cosA:
        #     - pick highest-pT muon in upper (phi>0) and lower (phi<0) hemisphere
        #     - REQUIRE timeNDof > NDOF_MIN on BOTH before trusting the time
        #     - dt = t_upper - t_lower ; veto if dt < -20 ns
        # ══════════════════════════════════════════════════════════════════
        survives_full = survives_cosA.copy()

        if np.sum(survives_cosA) > 0:
            mu = sorted_muons[ak.Array(survives_cosA)]          # cosA survivors
            up = mu[mu.phi > 0]
            lo = mu[mu.phi < 0]
            has_both = (ak.num(up) >= 1) & (ak.num(lo) >= 1)    # need a muon in each hemisphere

            if ak.sum(has_both) > 0:
                up_b = up[has_both]
                lo_b = lo[has_both]
                # subsets are slices of the pT-sorted collection -> [:, 0] is highest pT
                up_lead = up_b[:, 0]
                lo_lead = lo_b[:, 0]

                self.output["n_events_pair_present"] += int(ak.sum(has_both))

                # ndof on the two Delta-t muons
                up_ndof = up_lead.timeNDof
                lo_ndof = lo_lead.timeNDof
                ndof_ok = (up_ndof > NDOF_MIN) & (lo_ndof > NDOF_MIN)   # both have good timing

                # which of the two is the leading-pT (= the event's overall leading muon)?
                lead_is_upper = up_lead.pt >= lo_lead.pt
                lead_ndof = ak.where(lead_is_upper, up_ndof, lo_ndof)
                sub_ndof  = ak.where(lead_is_upper, lo_ndof, up_ndof)

                self.output["ndof_lead"].fill(val=ak.to_numpy(lead_ndof))
                self.output["ndof_sub"].fill(val=ak.to_numpy(sub_ndof))

                self.output["n_leadmu_ndof_pass"] += int(ak.sum(lead_ndof > NDOF_MIN))
                self.output["n_leadmu_ndof_fail"] += int(ak.sum(lead_ndof <= NDOF_MIN))
                self.output["n_submu_ndof_pass"]  += int(ak.sum(sub_ndof > NDOF_MIN))
                self.output["n_submu_ndof_fail"]  += int(ak.sum(sub_ndof <= NDOF_MIN))

                # events whose pair is removed from the dt veto because ndof fails
                self.output["n_events_excluded_ndof"] += int(ak.sum(~ndof_ok))

                # Delta t
                dt_all = up_lead.timeAtIpInOut - lo_lead.timeAtIpInOut

                # OLD/buggy behaviour: dt < -20 with no ndof requirement
                self.output["n_dt_veto_no_ndof"] += int(ak.sum(dt_all < -20))

                # CORRECT veto: ndof>7 on both AND dt < -20
                dt_veto = ndof_ok & (dt_all < -20)
                self.output["n_events_vetoed_ndof_dt"] += int(ak.sum(dt_veto))

                # Delta t of the events the cut actually acts on (ndof-passing)
                self.output["dt_ndofpass"].fill(val=ak.to_numpy(dt_all[ndof_ok]))

                # ── Count dt pairs: centered at zero vs the 25 ns satellite peaks ──
                _adt = ak.to_numpy(np.abs(dt_all[ndof_ok]))
                self.output["n_dt_ndofok"]  += int(_adt.size)
                self.output["n_dt_central"] += int(np.sum(_adt < 12.5))
                self.output["n_dt_bx1"]     += int(np.sum((_adt >= 12.5) & (_adt < 37.5)))
                self.output["n_dt_bx2"]     += int(np.sum((_adt >= 37.5) & (_adt < 62.5)))
                self.output["n_dt_bxhi"]    += int(np.sum(_adt >= 62.5))

                # ──────────────────────────────────────────────────────────
                # MUON TYPE (mutually exclusive) of the two dt muons, ONLY for
                # the events that the dt < -20 ns veto removes.  Leading should
                # be Global (it passed mediumId); we want to see whether the
                # subleading is "Standalone only" (-> would be killed by Loose ID).
                # Loose ID = isGlobal || isTracker  ; SA-only fails it.
                # ──────────────────────────────────────────────────────────
                lead_g = _as_bool(ak.where(lead_is_upper, up_lead.isGlobal,     lo_lead.isGlobal))
                lead_t = _as_bool(ak.where(lead_is_upper, up_lead.isTracker,    lo_lead.isTracker))
                lead_s = _as_bool(ak.where(lead_is_upper, up_lead.isStandalone, lo_lead.isStandalone))
                sub_g  = _as_bool(ak.where(lead_is_upper, lo_lead.isGlobal,     up_lead.isGlobal))
                sub_t  = _as_bool(ak.where(lead_is_upper, lo_lead.isTracker,    up_lead.isTracker))
                sub_s  = _as_bool(ak.where(lead_is_upper, lo_lead.isStandalone, up_lead.isStandalone))

                # ── 2D timeAtIpInOut: leading-pT (x) vs subleading-pT (y) muon of the
                #    Delta-t pair, split by the SUBLEADING muon's type.  ONLY events with
                #    ndof > 7 on BOTH dt muons (ndof_ok) -- the population the Delta t cut
                #    actually acts on; timing is not trusted otherwise.
                #    Types mutually exclusive: Global > Tracker(not global) > SA-only.
                lead_time = ak.where(lead_is_upper, up_lead.timeAtIpInOut, lo_lead.timeAtIpInOut)
                sub_time  = ak.where(lead_is_upper, lo_lead.timeAtIpInOut, up_lead.timeAtIpInOut)

                sub_is_global  = sub_g & ndof_ok
                sub_is_tracker = sub_t & ~sub_g & ndof_ok
                sub_is_saonly  = sub_s & ~sub_g & ~sub_t & ndof_ok

                self.output["time_lead_vs_sub_global"].fill(
                    lead=ak.to_numpy(lead_time[sub_is_global]),
                    sub =ak.to_numpy(sub_time[sub_is_global]))
                self.output["time_lead_vs_sub_tracker"].fill(
                    lead=ak.to_numpy(lead_time[sub_is_tracker]),
                    sub =ak.to_numpy(sub_time[sub_is_tracker]))
                self.output["time_lead_vs_sub_standalone"].fill(
                    lead=ak.to_numpy(lead_time[sub_is_saonly]),
                    sub =ak.to_numpy(sub_time[sub_is_saonly]))

                # ── Investigative: compare the LEADING muon to the SUBLEADING muon,
                #    the latter split by its timing.  Three categories per variable:
                #      "leading"                  -> leading muon (all ndof_ok pairs)
                #      "subleading (in-time)"     -> subleading, |t| < 12.5 ns
                #      "subleading (out-of-time)" -> subleading, |t| >= 12.5 ns
                #    The leading muon is in-time by construction, so it is NOT split. ──
                lead_pt    = ak.where(lead_is_upper, up_lead.pt,                lo_lead.pt)
                lead_eta   = ak.where(lead_is_upper, up_lead.eta,               lo_lead.eta)
                lead_phi   = ak.where(lead_is_upper, up_lead.phi,               lo_lead.phi)
                lead_dxy   = ak.where(lead_is_upper, up_lead.dxy,               lo_lead.dxy)
                lead_dz    = ak.where(lead_is_upper, up_lead.dz,                lo_lead.dz)
                lead_terr  = ak.where(lead_is_upper, up_lead.timeAtIpInOutErr,  lo_lead.timeAtIpInOutErr)
                lead_dxybs = ak.where(lead_is_upper, up_lead.dxybs,             lo_lead.dxybs)

                sub_pt    = ak.where(lead_is_upper, lo_lead.pt,                up_lead.pt)
                sub_eta   = ak.where(lead_is_upper, lo_lead.eta,               up_lead.eta)
                sub_phi   = ak.where(lead_is_upper, lo_lead.phi,               up_lead.phi)
                sub_dxy   = ak.where(lead_is_upper, lo_lead.dxy,               up_lead.dxy)
                sub_dz    = ak.where(lead_is_upper, lo_lead.dz,                up_lead.dz)
                sub_terr  = ak.where(lead_is_upper, lo_lead.timeAtIpInOutErr,  up_lead.timeAtIpInOutErr)
                sub_dxybs = ak.where(lead_is_upper, lo_lead.dxybs,             up_lead.dxybs)

                sub_intime  = ndof_ok & (np.abs(sub_time) <  12.5)
                sub_offtime = ndof_ok & (np.abs(sub_time) >= 12.5)

                # subleading time (shows the comb + where the 12.5 ns split sits)
                self.output["sub_time"].fill(val=ak.to_numpy(sub_time[ndof_ok]))

                # per-variable 3-curve overlay
                _props = {
                    "prop_pt":     (lead_pt,    sub_pt),
                    "prop_eta":    (lead_eta,   sub_eta),
                    "prop_dxy":    (lead_dxy,   sub_dxy),
                    "prop_dz":     (lead_dz,    sub_dz),
                    "prop_dxybs":  (lead_dxybs, sub_dxybs),
                    "prop_timeErr":(lead_terr,  sub_terr),
                }
                for _name, (_la, _sa) in _props.items():
                    self.output[_name].fill(tcat="leading",                  val=ak.to_numpy(_la[ndof_ok]))
                    self.output[_name].fill(tcat="subleading (in-time)",     val=ak.to_numpy(_sa[sub_intime]))
                    self.output[_name].fill(tcat="subleading (out-of-time)", val=ak.to_numpy(_sa[sub_offtime]))

                # cosα(lead, sub): one value per pair, overlaid in-time vs out-of-time
                _dot_ls = up_lead.px * lo_lead.px + up_lead.py * lo_lead.py + up_lead.pz * lo_lead.pz
                _mag_ls = up_lead.p * lo_lead.p
                cosA_ls = ak.where(_mag_ls > 0, _dot_ls / _mag_ls, -2.0)
                self.output["cosA_lead_sub_by_time"].fill(tcat="in-time",     val=ak.to_numpy(cosA_ls[sub_intime]))
                self.output["cosA_lead_sub_by_time"].fill(tcat="out-of-time", val=ak.to_numpy(cosA_ls[sub_offtime]))

                # 2D leading vs subleading (eta, pt, phi), split by sub in/out-of-time
                for _tc, _mm in (("in-time", sub_intime), ("out-of-time", sub_offtime)):
                    self.output["eta2d_lead_sub"].fill(tcat=_tc, lead=ak.to_numpy(lead_eta[_mm]), sub=ak.to_numpy(sub_eta[_mm]))
                    self.output["pt2d_lead_sub"].fill( tcat=_tc, lead=ak.to_numpy(lead_pt[_mm]),  sub=ak.to_numpy(sub_pt[_mm]))
                    self.output["phi2d_lead_sub"].fill(tcat=_tc, lead=ak.to_numpy(lead_phi[_mm]), sub=ak.to_numpy(sub_phi[_mm]))

                # restrict to dt-vetoed events
                lg, lt, ls = lead_g[dt_veto], lead_t[dt_veto], lead_s[dt_veto]
                sg, st, ss = sub_g[dt_veto],  sub_t[dt_veto],  sub_s[dt_veto]

                # leading dt muon (priority: Global > Tracker-not-global > SA-only > None)
                self.output["n_dtveto_lead_global"]  += int(ak.sum(lg))
                self.output["n_dtveto_lead_tracker"] += int(ak.sum(lt & ~lg))
                self.output["n_dtveto_lead_saonly"]  += int(ak.sum(ls & ~lg & ~lt))
                self.output["n_dtveto_lead_none"]    += int(ak.sum(~lg & ~lt & ~ls))
                # subleading dt muon
                self.output["n_dtveto_sub_global"]   += int(ak.sum(sg))
                self.output["n_dtveto_sub_tracker"]  += int(ak.sum(st & ~sg))
                self.output["n_dtveto_sub_saonly"]   += int(ak.sum(ss & ~sg & ~st))
                self.output["n_dtveto_sub_none"]     += int(ak.sum(~sg & ~st & ~ss))

                # map the veto back to the full event array
                idx_cosA = np.where(survives_cosA)[0]
                idx_pair = idx_cosA[ak.to_numpy(has_both)]
                veto_idx = idx_pair[ak.to_numpy(dt_veto)]
                survives_full[veto_idx] = False

        self.output["n_numer_full"] += int(np.sum(survives_full))
        self.output["lead_pt_post_full"].fill(val=lead.pt[ak.Array(survives_full)])

        return self.output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    import glob
    import re

    SAMPLE_DIR = "samples/Signal_Samples"
    OUTPUT_DIR = "cosA_signal_check_plots"
    PREFIX     = "cosA_check_"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*_preprocessed.pkl")))
    print(f"Found {len(pkl_files)} signal samples in {SAMPLE_DIR}/\n")

    executor = processor.FuturesExecutor(workers=8)
    runner   = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
        skipbadfiles=True,
    )

    for pkl_path in pkl_files:
        basename = os.path.basename(pkl_path).replace("_preprocessed.pkl", "")

        match = re.match(r"Stau_(\d+)_(\d+)(mm)", basename)
        if match:
            mass, lifetime, unit = match.groups()
            title_suffix = rf"(Stau {mass} GeV {lifetime} {unit})"
        else:
            title_suffix = f"({basename})"

        print(f"\n{'#' * 60}")
        print(f"  Processing: {basename}")
        print(f"{'#' * 60}")

        with open(pkl_path, "rb") as f:
            runnable = pickle.load(f)

        out = runner(
            runnable,
            treename="Events",
            processor_instance=CosASignalCheckProcessor(),
        )

        denom = out["n_signal_denom"]

        def eff(n):
            return f"{n / denom * 100:.4f}%" if denom > 0 else "n/a"

        print("\n" + "=" * 64)
        print(f"  {basename}")
        print("=" * 64)
        print("  EVENT COUNTS")
        print(f"    1. After signal selection (jet+MET+>=1 mu):  {out['n_after_jetmet_mu']}")
        print(f"       After leading-muon quality cuts (denom):  {denom}")
        print(f"    2. Events with 2+ muons:                     {out['n_events_2plus_muons']}")
        print(f"    3. Events with 2+ muons (post duplicate rm):  {out['n_events_2plus_muons_postdup']}"
              f"   (dup tracks removed: {out['n_duplicate_tracks_removed']})")
        print(f"    4. Vetoed by cosA (< -0.99):                 {out['n_events_vetoed_cosA']}   ({eff(out['n_events_vetoed_cosA'])})")
        print(f"    5. Excluded from dt veto by ndof<=7:         {out['n_events_excluded_ndof']}")
        print(f"    6. Vetoed by ndof>7 AND dt<-20 ns:           {out['n_events_vetoed_ndof_dt']}   ({eff(out['n_events_vetoed_ndof_dt'])})")
        print()
        print("  IS IT NDOF OR DELTA-t THAT HURTS?  (events with an upper+lower pair)")
        print(f"    Pair-present events (post-cosA):             {out['n_events_pair_present']}")
        print(f"    dt < -20 ignoring ndof (OLD/buggy behaviour): {out['n_dt_veto_no_ndof']}")
        print(f"    dt < -20 WITH ndof>7 required (correct):      {out['n_events_vetoed_ndof_dt']}")
        print(f"    -> events SAVED by requiring ndof>7:          {out['n_dt_veto_no_ndof'] - out['n_events_vetoed_ndof_dt']}")
        print()
        print("  DOES ndof>7 REMOVE LEADING OR SUBLEADING MUONS?  (the two dt muons)")
        print(f"    Leading-pT  dt muon:  pass {out['n_leadmu_ndof_pass']:>6}   fail {out['n_leadmu_ndof_fail']:>6}")
        print(f"    Subleading  dt muon:  pass {out['n_submu_ndof_pass']:>6}   fail {out['n_submu_ndof_fail']:>6}")
        print()
        print("  MUON TYPE IN dt-VETOED EVENTS  (mutually exclusive; one muon per event per row)")
        print(f"    {'':22}{'Global':>9}{'Trk(noG)':>10}{'SA-only':>9}{'None':>7}")
        print(f"    {'Leading dt muon':22}"
              f"{out['n_dtveto_lead_global']:>9}{out['n_dtveto_lead_tracker']:>10}"
              f"{out['n_dtveto_lead_saonly']:>9}{out['n_dtveto_lead_none']:>7}")
        print(f"    {'Subleading dt muon':22}"
              f"{out['n_dtveto_sub_global']:>9}{out['n_dtveto_sub_tracker']:>10}"
              f"{out['n_dtveto_sub_saonly']:>9}{out['n_dtveto_sub_none']:>7}")
        print("    (SA-only = isStandalone & !isGlobal & !isTracker -> FAILS Loose ID = isGlobal||isTracker)")
        print()
        print("  EFFICIENCY (numer / denom)")
        print(f"    Survive cosA:                 {eff(out['n_numer_cosA'])}")
        print(f"    Survive cosA + ndof + dt:     {eff(out['n_numer_full'])}")
        print("=" * 64 + "\n")

        # ── dt (upper - lower) BX structure: centered at zero vs 25 ns satellites ──
        n_dt_tot = out["n_dt_ndofok"]
        def _pct(n):
            return f"{n / n_dt_tot * 100:.2f}%" if n_dt_tot > 0 else "n/a"
        _sat = out["n_dt_bx1"] + out["n_dt_bx2"] + out["n_dt_bxhi"]
        n_gen = out["n_gen_signal"]
        n_sig = out["n_signal_denom"]          # events after the full signal selection (= eff() denom)
        def _pden(n):
            return f"{n / n_sig * 100:.4f}%" if n_sig > 0 else "n/a"
        print("  dt (upper - lower) for ndof>7 pairs: centered-at-zero vs 25 ns satellites")
        print(f"    gen mu+tau events (acceptance ref):  {n_gen}")
        print(f"    signal-selected events (denom):      {n_sig}")
        print(f"    total ndof>7 pairs:                  {n_dt_tot}  ({_pden(n_dt_tot)} of denom)")
        print(f"    |dt| < 12.5 ns   (centered at 0):    {out['n_dt_central']}  ({_pct(out['n_dt_central'])} of pairs)")
        print(f"    |dt| ~ 25 ns     (1 BX off):         {out['n_dt_bx1']}  ({_pct(out['n_dt_bx1'])} of pairs)")
        print(f"    |dt| ~ 50 ns     (2 BX off):         {out['n_dt_bx2']}  ({_pct(out['n_dt_bx2'])} of pairs)")
        print(f"    |dt| >= 62.5 ns  (>=3 BX off):       {out['n_dt_bxhi']}  ({_pct(out['n_dt_bxhi'])} of pairs)")
        print(f"    --> 25 ns satellites (|dt|>=12.5):   {_sat}  ({_pct(_sat)} of pairs)")
        print(f"  relative to signal-selected events (denom = {n_sig}):")
        print(f"    centered-at-0 / denom:               {out['n_dt_central']}/{n_sig} = {_pden(out['n_dt_central'])}")
        print(f"    25 ns satellites / denom:            {_sat}/{n_sig} = {_pden(_sat)}")
        print("=" * 64 + "\n")

        # ── Efficiency vs leading pT ──
        plot_cosA_efficiency(
            out["lead_pt_all"], out["lead_pt_post_cosA"],
            OUTPUT_DIR, PREFIX, title_suffix, basename + "_cosA",
            title=r"$\cos\alpha < -0.99$ veto efficiency",
        )
        plot_cosA_efficiency(
            out["lead_pt_all"], out["lead_pt_post_full"],
            OUTPUT_DIR, PREFIX, title_suffix, basename + "_full",
            title=r"$\cos\alpha$ + (ndof>7 & $\Delta t<-20$) efficiency",
        )

        # ── ndof of the two dt muons (leading vs subleading) ──
        plot_ndof_lead_sub(
            out["ndof_lead"], out["ndof_sub"],
            OUTPUT_DIR, PREFIX, title_suffix, basename,
        )

        # ── Delta t of ndof-passing events ──
        plot_dt(
            out["dt_ndofpass"],
            OUTPUT_DIR, PREFIX, title_suffix, basename,
        )

        # ── Muon type of the two dt muons, ONLY for dt-vetoed events ──
        plot_dtveto_muon_types(
            [out["n_dtveto_lead_global"], out["n_dtveto_lead_tracker"],
             out["n_dtveto_lead_saonly"], out["n_dtveto_lead_none"]],
            [out["n_dtveto_sub_global"], out["n_dtveto_sub_tracker"],
             out["n_dtveto_sub_saonly"], out["n_dtveto_sub_none"]],
            OUTPUT_DIR, PREFIX, title_suffix, basename,
        )

        # ── 2D timeAtIpInOut: leading vs subleading, split by subleading muon type ──
        plot_time2d(out["time_lead_vs_sub_global"],
                    r"Lead vs sub timeAtIpInOut — subleading GLOBAL " + title_suffix,
                    OUTPUT_DIR, PREFIX, basename + "_subGlobal")
        plot_time2d(out["time_lead_vs_sub_tracker"],
                    r"Lead vs sub timeAtIpInOut — subleading TRACKER (not global) " + title_suffix,
                    OUTPUT_DIR, PREFIX, basename + "_subTracker")
        plot_time2d(out["time_lead_vs_sub_standalone"],
                    r"Lead vs sub timeAtIpInOut — subleading STANDALONE-only " + title_suffix,
                    OUTPUT_DIR, PREFIX, basename + "_subStandalone")

        # ── Subleading-muon timeAtIpInOut (ndof>7): shows the 25 ns comb ──
        plot_sub_time(out["sub_time"], OUTPUT_DIR, PREFIX, basename + "_sub_time",
                      "Subleading muon timeAtIpInOut " + title_suffix)

        # ── Per-variable overlay: leading vs subleading(in-time) vs subleading(out-of-time) ──
        for _var, _tag in (("prop_pt", "pt"), ("prop_eta", "eta"), ("prop_dxy", "dxy"),
                           ("prop_dz", "dz"), ("prop_dxybs", "dxybs"), ("prop_timeErr", "timeErr")):
            plot_sub_overlay(out[_var], OUTPUT_DIR, PREFIX, basename + "_" + _tag,
                             "lead vs sub in/out-of-time " + title_suffix)

        # ── cosα(lead, sub), in-time vs out-of-time ──
        plot_sub_overlay(out["cosA_lead_sub_by_time"], OUTPUT_DIR, PREFIX,
                         basename + "_cosA_lead_sub",
                         r"$\cos\alpha$(lead, sub): in-time vs out-of-time " + title_suffix)

        # ── 2D leading vs subleading (eta, pt, phi), split by in/out-of-time ──
        plot_2d_by_time(out["eta2d_lead_sub"], OUTPUT_DIR, PREFIX, basename + "_eta2d",
                        "Lead vs sub eta " + title_suffix)
        plot_2d_by_time(out["pt2d_lead_sub"],  OUTPUT_DIR, PREFIX, basename + "_pt2d",
                        "Lead vs sub pT " + title_suffix)
        plot_2d_by_time(out["phi2d_lead_sub"], OUTPUT_DIR, PREFIX, basename + "_phi2d",
                        "Lead vs sub phi " + title_suffix)

    print("\nAll samples done!")