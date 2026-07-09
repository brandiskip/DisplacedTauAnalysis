#!/usr/bin/env python
"""
signal_muon_selection_study.py

Standalone study: how should the signal DisMuon be chosen in each event?

The three methods compared
--------------------------
  Method A: apply the quality cuts to ALL DisMuons, DELETE the ones that
            fail, and take the highest-pT survivor as the signal muon.
  Method B: take the highest-pT DisMuon first; keep the event only if THAT
            muon passes all the quality cuts (current approach).
  Method C: take the highest-pT DisMuon that passes all the quality cuts as
            the signal muon, but KEEP the failing muons in the event so the
            cosmic-ray veto can still see them.

  Methods A and C always choose the SAME muon; they differ only in whether
  the failing muons are deleted (A) or kept (C).  Deleting them matters for
  cosmic-ray studies: a deleted muon can be the second leg of a cosmic ray,
  and once deleted the cosmic veto cannot use it.

How we know which DisMuon is the real signal muon ("the true muon")
-------------------------------------------------------------------
Events are first required to have, at generator level, exactly 1 hadronic
tau and exactly 1 muon from the stau decay (and no electron).  That one
GenMuon is the signal muon by definition -- NO quality cuts are ever
applied to GenMuons.

Each reconstructed DisMuon is then matched to that GenMuon at the second
muon station (MB2), using the eta_at_mb2 / phi_at_mb2 values stored in the
n-tuples for both collections.  The DisMuon closest in DeltaR at MB2
(within DR_MATCH) is called "the true muon" throughout the output: it is
the reconstructed version of the generator-level signal muon.

Matching at the muon station instead of at the vertex is important for
displaced muons: their direction at the beamline does not have to agree
with the generator values, but the position where they cross the muon
system does.

Missing MB2 values
------------------
Some muons (Gen and reco) have no stored eta/phi_at_mb2 (NaN or sentinel).
A muon without MB2 values can never match, so the script counts them and
histograms their eta / phi / pt against muons WITH valid MB2 values
(the *_mb2_missing_* plots).  If the missing ones cluster at large |eta|,
the propagation is only filled for muons that reach the barrel station 2
cylinder, and endcap-going muons are simply not propagated.

What the quality cuts are (applied to DisMuons only)
----------------------------------------------------
  pt > 30 GeV, |eta| < 2.4, mediumId, pfRelIso03_all < 0.18,
  0.1 cm < |dxy| < 10 cm
These are the signal-region muon cuts of the analysis.  The study measures,
for each method, how often the muon the method selects is the true muon.

Run:  python signal_muon_selection_study.py
(needs the usual coffea environment and the *_preprocessed.pkl inputs)
"""

import os
import glob
import pickle
import re

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

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════
SAMPLE_DIR = "samples/Signal_Samples"
OUTPUT_DIR = "muon_selection_study_plots"
PREFIX     = "musel_"

# signal-muon quality cuts under study (the full set, applied together)
PT_MIN     = 30.0
ABSETA_MAX = 2.4
ISO_MAX    = 0.18
DXY_MIN    = 0.1     # cm
DXY_MAX    = 10.0    # cm

# gen matching
MATCH_MODE = "mb2"     # "mb2" = match at muon station 2 ; "vertex" = raw eta/phi
DR_MATCH   = 0.4       # match cone -- check the min-dR plots before trusting it
# (both DisMuon and GenPart carry precomputed eta_at_mb2/phi_at_mb2 branches;
#  the matching uses those stored values directly)

COSA_B2B   = -0.99     # cos(angle) below which two muons count as back-to-back

# (key, pretty label) for the per-cut breakdown -- order matters for the table
CUT_DEFS = [
    ("pt",     "pt>30"),
    ("eta",    "|eta|<2.4"),
    ("id",     "mediumId"),
    ("iso",    "iso<0.18"),
    ("dxymin", "|dxy|>0.1"),
    ("dxymax", "|dxy|<10"),
]


# ════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════
def _as_bool(x):
    """Cast a NanoAOD flag (bool or int8 0/1) to a real boolean awkward array."""
    return ak.values_astype(x, np.bool_)


def _wrap_phi(p):
    return (p + np.pi) % (2.0 * np.pi) - np.pi


def get_Lxy(genvistau):
    """Transverse decay length: tau production vertex (= stau decay vertex)
    minus the stau production vertex (= PV)."""
    vx = genvistau.parent.vx - genvistau.parent.distinctParent.vx
    vy = genvistau.parent.vy - genvistau.parent.distinctParent.vy
    return np.sqrt(vx ** 2 + vy ** 2)


def at_index(arr, opt_idx):
    """arr[i] per event, where opt_idx is an option-typed per-event index
    (None -> None).  Works on any jagged per-muon array."""
    return ak.firsts(arr[ak.singletons(opt_idx)])


def delta_r(eta1, phi1, eta2, phi2):
    dphi = _wrap_phi(phi1 - phi2)
    deta = eta1 - eta2
    return np.sqrt(deta ** 2 + dphi ** 2)


def mb2_missing(eta2, phi2):
    """True where the stored station-2 propagation is unusable: NaN/inf, or an
    obvious sentinel value far outside the physical range (|eta| or |phi| > 10,
    e.g. -99 or 999 fillers)."""
    bad = ~np.isfinite(eta2) | ~np.isfinite(phi2)
    bad = bad | (np.abs(eta2) > 10.0) | (np.abs(phi2) > 10.0)
    return bad


# ════════════════════════════════════════════════════════════════════
#  plotting
# ════════════════════════════════════════════════════════════════════
def plot_multi_eff(h_denom, numers, xlabel, outpath, title, denom_label="denominator"):
    """Top: denominator + numerator distributions.  Bottom: one efficiency curve
    per numerator (binomial errors).  `numers` = [(hist, label, color), ...]."""
    dvals = h_denom.values()
    if np.sum(dvals) == 0:
        print(f"    Skipping {outpath} (empty denominator)")
        return
    edges   = h_denom.axes["val"].edges
    centers = h_denom.axes["val"].centers

    fig, (ax_d, ax_e) = plt.subplots(
        2, 1, figsize=(8, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    ax_d.step(edges[:-1], dvals, where="post", color="black",
              linestyle="--", label=denom_label)
    for h, lab, col in numers:
        ax_d.step(edges[:-1], h.values(), where="post", color=col, label=lab)
    ax_d.set_ylabel("Events")
    ax_d.set_title(title)
    ax_d.legend(fontsize=9)

    for h, lab, col in numers:
        nvals = h.values()
        eff = np.where(dvals > 0, nvals / np.clip(dvals, 1, None), np.nan)
        err = np.where(dvals > 0,
                       np.sqrt(np.clip(eff * (1 - np.clip(eff, 0, 1)), 0, None)
                               / np.clip(dvals, 1, None)),
                       np.nan)
        ax_e.errorbar(centers, eff, yerr=err, fmt="o-", color=col,
                      capsize=2, markersize=3, label=lab)
    ax_e.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax_e.set_xlabel(xlabel)
    ax_e.set_ylabel("Efficiency")
    ax_e.set_ylim(-0.05, 1.15)
    ax_e.legend(fontsize=8)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {outpath}")


def plot_mindr(h, outpath, title):
    if np.sum(h.values()) == 0:
        print(f"    Skipping {outpath} (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = h.axes["val"].edges
    for cat in h.axes["tcat"]:
        vals = h[{"tcat": cat}].values()
        if vals.sum() == 0:
            continue
        ax.step(edges[:-1], vals, where="post", label=f"{cat}  (N={int(vals.sum())})")
    ax.axvline(DR_MATCH, color="red", ls="--", label=f"match cone dR < {DR_MATCH}")
    ax.set_yscale("log")
    ax.set_xlabel(r"min $\Delta R$(GenMuon, DisMuon)"
                  "\n(one entry per event; last bin = everything $\\geq$ 3,"
                  " incl. muons without valid MB2 values)")
    ax.set_ylabel("Events")
    ax.set_title(title)
    ax.legend()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {outpath}")


def plot_simple(h, outpath, title, xlabel=None, color="teal", vline=None):
    if np.sum(h.values()) == 0:
        print(f"    Skipping {outpath} (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = h.axes["val"].edges
    ax.step(edges[:-1], h.values(), where="post", color=color)
    if vline is not None:
        ax.axvline(vline, color="red", ls="--")
    ax.set_xlabel(xlabel or h.axes["val"].label)
    ax.set_ylabel("Events")
    ax.set_title(title)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {outpath}")


def plot_overlay(h, outpath, title, normalize=False):
    if np.sum(h.values()) == 0:
        print(f"    Skipping {outpath} (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = h.axes["val"].edges
    for cat in h.axes["tcat"]:
        vals = h[{"tcat": cat}].values()
        tot = vals.sum()
        if tot == 0:
            continue
        y = vals / tot if normalize else vals
        ax.step(edges[:-1], y, where="post", label=f"{cat}  (N={int(tot)})")
    ax.set_xlabel(h.axes["val"].label)
    ax.set_ylabel("Fraction / bin" if normalize else "Events / bin")
    ax.set_title(title)
    ax.legend()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {outpath}")


def plot_pt2d(h, outpath, title):
    if np.sum(h.values()) == 0:
        print(f"    Skipping {outpath} (empty)")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    h.plot2d(ax=ax, norm=mcolors.LogNorm(vmin=1))
    lo = 0
    hi = min(h.axes["lead"].edges[-1], h.axes["sub"].edges[-1])
    ax.plot([lo, hi], [lo, hi], color="white", lw=0.8, ls=":")
    ax.set_title(title)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {outpath}")


def plot_cut_bars(out, outpath, title):
    """Grouped bars: fraction of highest-pT muons / true muons failing each
    quality cut ('any' = fails that cut; 'only' = fails ONLY that cut)."""
    n_lead = out["n_after_jetmet_mu"]
    n_gm   = out["n_gm"]
    if n_lead == 0:
        print(f"    Skipping {outpath} (no events)")
        return
    labels = [lab for _, lab in CUT_DEFS]
    f_lead  = [out[f"cf_lead_{k}"] / n_lead for k, _ in CUT_DEFS]
    f_gm    = [(out[f"cf_gm_{k}"] / n_gm) if n_gm > 0 else 0 for k, _ in CUT_DEFS]
    o_lead  = [out[f"co_lead_{k}"] / n_lead for k, _ in CUT_DEFS]
    o_gm    = [(out[f"co_gm_{k}"] / n_gm) if n_gm > 0 else 0 for k, _ in CUT_DEFS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(labels))
    w = 0.38
    for ax, fl, fg, sub in ((ax1, f_lead, f_gm, "fails this cut"),
                            (ax2, o_lead, o_gm, "fails ONLY this cut")):
        ax.bar(x - w / 2, fl, w, label="highest-pT DisMuon", color="#4C72B0")
        ax.bar(x + w / 2, fg, w, label="true muon (matched to GenMuon)", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(sub)
        ax.legend(fontsize=9)
    ax1.set_ylabel("Fraction of muons")
    fig.suptitle(title)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {outpath}")


# ════════════════════════════════════════════════════════════════════
#  processor
# ════════════════════════════════════════════════════════════════════
class MuonSelectionStudyProcessor(processor.ProcessorABC):
    def __init__(self):
        out = {
            # ── event flow ──────────────────────────────────────────
            "n_events_initial":  0,
            "n_gen_signal":      0,   # 1 GenVisStauTau + 1 GenMuon + 0 GenElectron
            "n_after_jetmet_mu": 0,   # + jet==1 & MET>105 & >=1 DisMuon  == "selected events"
            "n_dup_removed":     0,   # duplicate DisMuon tracks removed
            "n_mu1": 0, "n_mu2": 0, "n_mu3p": 0,   # DisMuon multiplicity in selected events
            "n_2pluspass":       0,   # events with >=2 muons passing ALL cuts

            # ── finding the true muon (gen matching) ────────────────
            "n_gm":        0,   # selected events where the true muon was found
            "n_no_gm":     0,   # selected events where NO DisMuon matches the GenMuon
            "n_gm_multi":  0,   # >=2 DisMuons inside the match cone (double-counted tracks)
            "n_gm_flavok": 0,   # true muon also flagged by NanoAOD genPartFlav != 0
            "n_gm_lead":   0,   # true muon IS the highest-pT DisMuon
            "n_gm_sub":    0,   # true muon is NOT the highest-pT DisMuon

            # ── missing MB2 values (why some muons can never match) ─
            "n_gen_mb2_missing":  0,   # events whose GenMuon has no usable mb2 values
            "n_mu_total":         0,   # all DisMuons in selected events
            "n_mu_mb2_missing":   0,   # DisMuons with no usable mb2 values
            "n_event_all_mu_mb2_missing": 0,   # events where EVERY DisMuon misses mb2
                                               # (these pile up in the last min-dR bin)

            # ── best possible: true muon itself passes all cuts ─────
            "n_oracle": 0,

            # ── method B: highest-pT muon first, then cuts ──────────
            "nB_cand":        0,   # events kept (highest-pT muon passes all cuts)
            "nB_correct":     0,   # ... and that muon is the true muon
            "nB_wrong":       0,   # ... but the true muon is a different muon
            "nB_nomatch":     0,   # ... but no DisMuon in the event is the true muon
            "nB_lost_recov":  0,   # rejected, but the true muon passes cuts (avoidable loss)
            "nB_lost_unrec":  0,   # rejected, true muon fails cuts (lost with any method)
            "nB_nocand_nogm": 0,   # rejected, true muon never found in the event
            #    detail: when the highest-pT muon fails the cuts,
            "nB_faillead_leadgm":    0,   # ... the failing muon WAS the true muon
            "nB_faillead_gmsubpass": 0,   # ... a lower-pT true muon passes (same as lost_recov)

            # ── method A/C: highest-pT muon that passes all cuts ────
            "nAC_cand":          0,   # events kept (at least one muon passes all cuts)
            "nAC_correct":       0,   # ... chosen muon is the true muon
            "nAC_wrong_outrank": 0,   # ... true muon passes too, but a higher-pT muon also passes
            "nAC_wrong_gmfail":  0,   # ... true muon fails cuts, a different muon was chosen
            "nAC_nomatch":       0,   # ... no DisMuon in the event is the true muon
            "nAC_lost_gm":       0,   # rejected, true muon found but fails cuts
            "nAC_nocand_nogm":   0,   # rejected, true muon never found

            # ── difference between A and C ──────────────────────────
            "nA_prune_b2b": 0,  # events where a muon DELETED by method A is
                                # back-to-back with the chosen signal muon
        }
        # per-cut breakdown counters
        for k, _ in CUT_DEFS:
            out[f"cf_lead_{k}"] = 0   # highest-pT muon fails this cut
            out[f"co_lead_{k}"] = 0   # highest-pT muon fails ONLY this cut
            out[f"cf_gm_{k}"]   = 0   # true muon fails this cut
            out[f"co_gm_{k}"]   = 0   # true muon fails ONLY this cut

        # ── histograms ──────────────────────────────────────────────
        out["h_mindr"] = Hist(
            axis.StrCategory([], name="tcat", growth=True),
            axis.Regular(60, 0, 3, name="val", label=r"min $\Delta R$(GenMuon, DisMuon)"))
        out["h_gm_rank"] = Hist(
            axis.Regular(8, -0.5, 7.5, name="val",
                         label="pT rank of the true muon (0 = highest pT)"))
        out["h_nmu"] = Hist(
            axis.StrCategory([], name="tcat", growth=True),
            axis.Regular(10, -0.5, 9.5, name="val", label="DisMuons / event"))
        out["h_gmpt_by_rank"] = Hist(
            axis.StrCategory([], name="tcat", growth=True),
            axis.Regular(60, 0, 300, name="val", label=r"true muon $p_T$ [GeV]"))

        # eta / phi / pt of muons WITH vs WITHOUT usable MB2 values,
        # for DisMuons (per muon) and GenMuons (one per event)
        for coll in ("reco", "gen"):
            out[f"h_mb2_eta_{coll}"] = Hist(
                axis.StrCategory([], name="tcat", growth=True),
                axis.Regular(60, -3, 3, name="val",
                             label=(r"DisMuon $\eta$" if coll == "reco" else r"GenMuon $\eta$")))
            out[f"h_mb2_phi_{coll}"] = Hist(
                axis.StrCategory([], name="tcat", growth=True),
                axis.Regular(60, -np.pi, np.pi, name="val",
                             label=(r"DisMuon $\phi$" if coll == "reco" else r"GenMuon $\phi$")))
            out[f"h_mb2_pt_{coll}"] = Hist(
                axis.StrCategory([], name="tcat", growth=True),
                axis.Regular(60, 0, 300, name="val",
                             label=(r"DisMuon $p_T$ [GeV]" if coll == "reco" else r"GenMuon $p_T$ [GeV]")))

        for tag in ("evdenom", "denom", "B", "AC", "oracle"):
            out[f"h_pt_{tag}"] = Hist(
                axis.Regular(25, 0, 500, name="val", label=r"Gen muon $p_T$ [GeV]"))
            out[f"h_lxy_{tag}"] = Hist(
                axis.Regular(25, 0, 100, name="val", label=r"Gen muon production $L_{xy}$ [cm]"))

        out["h_pt2d_lead_gm"] = Hist(
            axis.Regular(50, 0, 500, name="lead", label=r"Highest-$p_T$ DisMuon $p_T$ [GeV]"),
            axis.Regular(50, 0, 200, name="sub",  label=r"true muon $p_T$ [GeV]"))

        self.output = out

    # ----------------------------------------------------------------
    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")
        self.output["n_events_initial"] += len(events)

        is_signal = not ("Cosmic" in dataset or dataset.startswith("LooseMu")
                         or dataset == "test_cosmics_calib" or "NoBPTX" in dataset)
        has_gen = "GenPart" in events.fields
        if not (is_signal and has_gen):
            return self.output

        # ════════════════════════════════════════════════════════════
        # GEN-LEVEL SIGNAL REGION (before any DisMuon cuts):
        #   exactly 1 GenVisStauTau, exactly 1 GenMuon, 0 GenElectrons
        # NOTE: GenMuons are only required to pass pt>20, |eta|<2.4 and to
        # come from a stau.  The quality cuts are NEVER applied to GenMuons.
        # ════════════════════════════════════════════════════════════
        gpart = events.GenPart

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

        events['GenMuon'] = gpart[(abs(gpart.pdgId) == 13) & gpart.hasFlags("isLastCopy")]
        events['GenMuon'] = events.GenMuon[
            (events.GenMuon.pt > 20) &
            (abs(events.GenMuon.eta) < 2.4) &
            (abs(events.GenMuon.distinctParent.distinctParent.pdgId) == 1000015)
        ]

        events['GenElectron'] = gpart[(abs(gpart.pdgId) == 11) & gpart.hasFlags("isLastCopy")]
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

        # ── DisMuon collection: NO per-muon cuts applied here.  Everything the
        #    methods need is carried along; the methods act later. ──
        events["DisMuon"] = ak.zip(
            {
                "pt":             events.DisMuon.pt,
                "eta":            events.DisMuon.eta,
                "phi":            events.DisMuon.phi,
                "eta_at_mb2":     events.DisMuon.eta_at_mb2,
                "phi_at_mb2":     events.DisMuon.phi_at_mb2,
                "mass":           events.DisMuon.mass,
                "charge":         events.DisMuon.charge,
                "dxy":            events.DisMuon.dxy,
                "dz":             events.DisMuon.dz,
                "mediumId":       events.DisMuon.mediumId,
                "pfRelIso03_all": events.DisMuon.pfRelIso03_all,
                "isGlobal":       events.DisMuon.isGlobal,
                "isTracker":      events.DisMuon.isTracker,
                "isStandalone":   events.DisMuon.isStandalone,
                "genPartFlav":    events.DisMuon.genPartFlav,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # sort by pT (descending) once; index 0 = highest pT everywhere below
        events["DisMuon"] = events.DisMuon[
            ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]

        # ── duplicate-track removal (same-charge, deta<0.01, dphi<0.001, dpt<0.5;
        #    keeps the first = highest-pT copy) ──
        muons_s = events.DisMuon
        local_i = ak.local_index(muons_s, axis=1)
        a, b   = ak.unzip(ak.cartesian([muons_s, muons_s], axis=1, nested=True))
        ia, ib = ak.unzip(ak.cartesian([local_i, local_i], axis=1, nested=True))
        match = (
            ((a.charge * b.charge) > 0)
            & (abs(a.eta - b.eta) < 0.01)
            & (abs(a.delta_phi(b)) < 0.001)
            & (abs(a.pt - b.pt) < 0.5)
            & (ib < ia)
        )
        is_dup = ak.any(match, axis=2)
        self.output["n_dup_removed"] += int(ak.sum(is_dup))
        events["DisMuon"] = muons_s[~is_dup]

        # ── event-level signal selection: 1 displaced jet + MET > 105 ──
        charged_sel = events.Jet.constituents.pf.charge != 0
        jet_dxy = ak.where(
            ak.all(events.Jet.constituents.pf.charge == 0, axis=-1),
            -999,
            ak.flatten(
                events.Jet.constituents.pf[
                    ak.argmax(events.Jet.constituents.pf[charged_sel].pt,
                              axis=2, keepdims=True)
                ].d0,
                axis=-1,
            ),
        )
        jet_dxy = ak.fill_none(jet_dxy, -999)
        events["Jet"] = ak.with_field(events.Jet, jet_dxy, where="dxy")

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
        signal_mask = (ak.num(jets) == 1) & (events.PFMET.pt > 105)
        events = events[signal_mask]
        if len(events) == 0:
            return self.output

        # ── "selected events": >=1 DisMuon (no muon quality cuts yet) ──
        events = events[ak.num(events.DisMuon) > 0]
        if len(events) == 0:
            return self.output
        n_ev = len(events)
        self.output["n_after_jetmet_mu"] += n_ev

        muons  = events.DisMuon
        n_mu   = ak.num(muons)
        self.output["n_mu1"]  += int(ak.sum(n_mu == 1))
        self.output["n_mu2"]  += int(ak.sum(n_mu == 2))
        self.output["n_mu3p"] += int(ak.sum(n_mu >= 3))
        self.output["h_nmu"].fill(tcat="all DisMuons", val=ak.to_numpy(n_mu))

        # exactly one GenMuon per event by construction
        gen_mu  = ak.firsts(events.GenMuon)
        gen_pt  = ak.to_numpy(gen_mu.pt)
        gen_eta = ak.to_numpy(gen_mu.eta)
        gen_phi = ak.to_numpy(gen_mu.phi)
        gen_lxy = ak.to_numpy(np.sqrt(gen_mu.vx ** 2 + gen_mu.vy ** 2))

        # ════════════════════════════════════════════════════════════
        # MISSING MB2 VALUES: which muons have no usable stored propagation?
        # Histogram eta/phi/pt of muons with vs without MB2 values -- if the
        # missing ones cluster at large |eta|, they are endcap-going muons
        # that never reach the barrel station-2 cylinder.
        # ════════════════════════════════════════════════════════════
        reco_missing = mb2_missing(muons.eta_at_mb2, muons.phi_at_mb2)
        gen_missing  = mb2_missing(gen_mu.eta_at_mb2, gen_mu.phi_at_mb2)
        gen_missing_np = ak.to_numpy(gen_missing)

        self.output["n_gen_mb2_missing"] += int(np.sum(gen_missing_np))
        self.output["n_mu_total"]        += int(ak.sum(ak.num(muons)))
        self.output["n_mu_mb2_missing"]  += int(ak.sum(reco_missing))
        self.output["n_event_all_mu_mb2_missing"] += int(ak.sum(ak.all(reco_missing, axis=1)))

        for var, arr in (("eta", muons.eta), ("phi", muons.phi), ("pt", muons.pt)):
            self.output[f"h_mb2_{var}_reco"].fill(
                tcat="MB2 values present",
                val=ak.to_numpy(ak.flatten(arr[~reco_missing])))
            self.output[f"h_mb2_{var}_reco"].fill(
                tcat="MB2 values missing",
                val=ak.to_numpy(ak.flatten(arr[reco_missing])))
        for var, arr in (("eta", gen_eta), ("phi", gen_phi), ("pt", gen_pt)):
            self.output[f"h_mb2_{var}_gen"].fill(
                tcat="MB2 values present", val=arr[~gen_missing_np])
            self.output[f"h_mb2_{var}_gen"].fill(
                tcat="MB2 values missing", val=arr[gen_missing_np])

        # ════════════════════════════════════════════════════════════
        # FIND THE TRUE MUON: match each DisMuon to the GenMuon at muon
        # station 2, using the stored eta_at_mb2/phi_at_mb2 of both.
        # The closest DisMuon within DR_MATCH is "the true muon".
        # ════════════════════════════════════════════════════════════
        dr_mb2 = delta_r(muons.eta_at_mb2, muons.phi_at_mb2,
                         gen_mu.eta_at_mb2, gen_mu.phi_at_mb2)
        # a muon (reco or gen) without usable MB2 values can never match
        dr_mb2 = ak.where(reco_missing | gen_missing, 999.0, dr_mb2)
        dr_mb2 = ak.where(np.isfinite(dr_mb2), dr_mb2, 999.0)
        dr_vtx = delta_r(muons.eta, muons.phi, gen_mu.eta, gen_mu.phi)

        # one entry per event: distance to the CLOSEST DisMuon.  Values beyond
        # the 0-3 axis (including the 999 no-valid-MB2 sentinel) are clipped
        # into the last bin so every selected event is visible on the plot.
        self.output["h_mindr"].fill(
            tcat="MB2",    val=np.minimum(ak.to_numpy(ak.min(dr_mb2, axis=1)), 2.99))
        self.output["h_mindr"].fill(
            tcat="vertex", val=np.minimum(ak.to_numpy(ak.min(dr_vtx, axis=1)), 2.99))

        dr = dr_mb2 if MATCH_MODE == "mb2" else dr_vtx
        min_dr  = ak.min(dr, axis=1)
        best    = ak.firsts(ak.argmin(dr, axis=1, keepdims=True))
        matched = ak.fill_none(min_dr < DR_MATCH, False)
        gm_idx  = ak.mask(best, matched)     # pT rank of the true muon (None = not found)
        has_gm  = ~ak.is_none(gm_idx)
        gm_f    = ak.to_numpy(ak.fill_none(gm_idx, -1))    # -1 = true muon not found
        has_gm_np = gm_f >= 0

        n_gm = int(np.sum(has_gm_np))
        self.output["n_gm"]       += n_gm
        self.output["n_no_gm"]    += int(n_ev - n_gm)
        self.output["n_gm_multi"] += int(ak.sum(ak.sum(dr < DR_MATCH, axis=1) >= 2))

        gm_flav = ak.to_numpy(ak.fill_none(at_index(muons.genPartFlav, gm_idx), 0))
        self.output["n_gm_flavok"] += int(np.sum(has_gm_np & (gm_flav != 0)))

        gm_lead_np = has_gm_np & (gm_f == 0)
        gm_sub_np  = has_gm_np & (gm_f > 0)
        self.output["n_gm_lead"] += int(np.sum(gm_lead_np))
        self.output["n_gm_sub"]  += int(np.sum(gm_sub_np))
        self.output["h_gm_rank"].fill(val=np.clip(gm_f[has_gm_np], 0, 7))

        gm_pt = ak.to_numpy(ak.fill_none(at_index(muons.pt, gm_idx), -1.0))
        self.output["h_gmpt_by_rank"].fill(tcat="true muon is highest-pT",
                                           val=gm_pt[gm_lead_np])
        self.output["h_gmpt_by_rank"].fill(tcat="true muon is NOT highest-pT",
                                           val=gm_pt[gm_sub_np])

        # ════════════════════════════════════════════════════════════
        # QUALITY CUTS (applied to DisMuons only) + per-cut bookkeeping
        # ════════════════════════════════════════════════════════════
        cut_masks = [
            ("pt",     muons.pt > PT_MIN),
            ("eta",    np.abs(muons.eta) < ABSETA_MAX),
            ("id",     _as_bool(muons.mediumId)),
            ("iso",    muons.pfRelIso03_all < ISO_MAX),
            ("dxymin", np.abs(muons.dxy) > DXY_MIN),
            ("dxymax", np.abs(muons.dxy) < DXY_MAX),
        ]
        pass_all = cut_masks[0][1]
        for _, m in cut_masks[1:]:
            pass_all = pass_all & m

        n_pass = ak.sum(pass_all, axis=1)
        self.output["h_nmu"].fill(tcat="passing ALL cuts", val=ak.to_numpy(n_pass))
        self.output["n_2pluspass"] += int(ak.sum(n_pass >= 2))

        # per-cut pass/fail of the highest-pT muon and of the true muon
        lead_by_cut = {k: ak.to_numpy(m[:, 0]) for k, m in cut_masks}
        gm_by_cut   = {k: ak.to_numpy(ak.fill_none(at_index(m, gm_idx), True))
                       for k, m in cut_masks}
        for k, _ in CUT_DEFS:
            oth_l = np.all([lead_by_cut[k2] for k2, _ in CUT_DEFS if k2 != k], axis=0)
            oth_g = np.all([gm_by_cut[k2]   for k2, _ in CUT_DEFS if k2 != k], axis=0)
            self.output[f"cf_lead_{k}"] += int(np.sum(~lead_by_cut[k]))
            self.output[f"co_lead_{k}"] += int(np.sum(~lead_by_cut[k] & oth_l))
            self.output[f"cf_gm_{k}"]   += int(np.sum(has_gm_np & ~gm_by_cut[k]))
            self.output[f"co_gm_{k}"]   += int(np.sum(has_gm_np & ~gm_by_cut[k] & oth_g))

        # ════════════════════════════════════════════════════════════
        # THE THREE METHODS
        # ════════════════════════════════════════════════════════════
        lead_pass = ak.to_numpy(ak.fill_none(ak.firsts(pass_all), False))

        idx        = ak.local_index(muons.pt, axis=1)
        first_pass = ak.firsts(idx[pass_all])     # highest-pT muon passing all cuts
        has_ac     = ~ak.is_none(first_pass)
        ac_f       = ak.to_numpy(ak.fill_none(first_pass, -2))
        has_ac_np  = ac_f >= 0

        gm_pass = ak.to_numpy(ak.fill_none(at_index(pass_all, gm_idx), False))

        # best possible: the true muon itself passes all cuts
        oracle_np = has_gm_np & gm_pass
        self.output["n_oracle"] += int(np.sum(oracle_np))

        # ── method B: highest-pT muon first, then cuts ──
        B_correct = lead_pass & (gm_f == 0)
        B_wrong   = lead_pass & has_gm_np & (gm_f != 0)
        B_nomatch = lead_pass & ~has_gm_np
        noB       = ~lead_pass
        self.output["nB_cand"]        += int(np.sum(lead_pass))
        self.output["nB_correct"]     += int(np.sum(B_correct))
        self.output["nB_wrong"]       += int(np.sum(B_wrong))
        self.output["nB_nomatch"]     += int(np.sum(B_nomatch))
        self.output["nB_lost_recov"]  += int(np.sum(noB & has_gm_np & gm_pass))
        self.output["nB_lost_unrec"]  += int(np.sum(noB & has_gm_np & ~gm_pass))
        self.output["nB_nocand_nogm"] += int(np.sum(noB & ~has_gm_np))
        self.output["nB_faillead_leadgm"]    += int(np.sum(noB & (gm_f == 0)))
        self.output["nB_faillead_gmsubpass"] += int(np.sum(noB & (gm_f > 0) & gm_pass))

        # ── method A/C: highest-pT muon that passes all cuts ──
        AC_correct = has_ac_np & has_gm_np & (ac_f == gm_f)
        AC_wrong   = has_ac_np & has_gm_np & (ac_f != gm_f)
        self.output["nAC_cand"]          += int(np.sum(has_ac_np))
        self.output["nAC_correct"]       += int(np.sum(AC_correct))
        self.output["nAC_wrong_outrank"] += int(np.sum(AC_wrong & gm_pass))
        self.output["nAC_wrong_gmfail"]  += int(np.sum(AC_wrong & ~gm_pass))
        self.output["nAC_nomatch"]       += int(np.sum(has_ac_np & ~has_gm_np))
        self.output["nAC_lost_gm"]       += int(np.sum(~has_ac_np & has_gm_np))
        self.output["nAC_nocand_nogm"]   += int(np.sum(~has_ac_np & ~has_gm_np))

        # ── difference between A and C: method A DELETES the muons that fail
        #    the cuts.  Count the events where a deleted muon is back-to-back
        #    with the chosen signal muon -- on cosmic/NoBPTX data such a muon
        #    can be the second leg of a cosmic ray, and after deletion the
        #    cosmic veto cannot use it.  Method C keeps these muons. ──
        cpx = ak.to_numpy(ak.fill_none(at_index(muons.px, first_pass), 0.0))
        cpy = ak.to_numpy(ak.fill_none(at_index(muons.py, first_pass), 0.0))
        cpz = ak.to_numpy(ak.fill_none(at_index(muons.pz, first_pass), 0.0))
        cp  = np.sqrt(cpx ** 2 + cpy ** 2 + cpz ** 2)
        removed = muons[~pass_all]
        dot = cpx * removed.px + cpy * removed.py + cpz * removed.pz
        mag = cp * removed.p
        cosr = ak.where(mag > 0, dot / mag, 0.0)
        b2b = ak.to_numpy(ak.fill_none(ak.any(cosr < COSA_B2B, axis=1), False))
        self.output["nA_prune_b2b"] += int(np.sum(b2b & has_ac_np))

        # ── 2D: highest-pT muon vs true muon pT, for events where the true
        #    muon is NOT the highest-pT one ──
        lead_pt = ak.to_numpy(muons.pt[:, 0])
        self.output["h_pt2d_lead_gm"].fill(
            lead=lead_pt[gm_sub_np], sub=gm_pt[gm_sub_np])

        # ── efficiency inputs vs gen muon pT and gen muon production Lxy ──
        for var, arr in (("pt", gen_pt), ("lxy", gen_lxy)):
            self.output[f"h_{var}_evdenom"].fill(val=arr)               # all selected events
            self.output[f"h_{var}_denom"].fill(val=arr[has_gm_np])      # true muon found
            self.output[f"h_{var}_B"].fill(val=arr[B_correct])          # B picks the true muon
            self.output[f"h_{var}_AC"].fill(val=arr[AC_correct])        # A/C pick the true muon
            self.output[f"h_{var}_oracle"].fill(val=arr[oracle_np])     # true muon passes cuts

        return self.output

    def postprocess(self, accumulator):
        return accumulator


# ════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*_preprocessed.pkl")))
    print(f"Found {len(pkl_files)} signal samples in {SAMPLE_DIR}/")
    print(f"Gen matching: mode={MATCH_MODE}, dR < {DR_MATCH} "
          f"(stored eta_at_mb2/phi_at_mb2 on DisMuon and GenPart)\n")

    executor = processor.FuturesExecutor(workers=8)
    runner   = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
        skipbadfiles=True,
    )

    for pkl_path in pkl_files:
        basename = os.path.basename(pkl_path).replace("_preprocessed.pkl", "")
        m = re.match(r"Stau_(\d+)_(\d+)(mm)", basename)
        title_suffix = (rf"(Stau {m.group(1)} GeV {m.group(2)} {m.group(3)})"
                        if m else f"({basename})")

        print(f"\n{'#' * 64}\n  Processing: {basename}\n{'#' * 64}")

        with open(pkl_path, "rb") as f:
            runnable = pickle.load(f)

        out = runner(
            runnable,
            treename="Events",
            processor_instance=MuonSelectionStudyProcessor(),
        )

        D    = out["n_after_jetmet_mu"]      # selected events
        n_gm = out["n_gm"]                   # events where the true muon was found

        def pct(n, d):
            return f"{n / d * 100:.2f}%" if d > 0 else "n/a"

        print("\n" + "=" * 76)
        print(f"  SAMPLE: {basename}")
        print(f"  (true muon = DisMuon matched to the GenMuon at muon station 2,")
        print(f"   match cone dR < {DR_MATCH}; quality cuts are applied to DisMuons only)")
        print("=" * 76)

        print()
        print("  STEP 1 -- EVENT SELECTION: how many events enter the study")
        print(f"    events in the sample:                              {out['n_events_initial']}")
        print(f"    with 1 gen tau + 1 gen muon from the stau decay:   {out['n_gen_signal']}")
        print(f"    also passing jet==1, MET>105, >=1 DisMuon:         {D}")
        print(f"    --> these {D} events are the 'selected events'; every number")
        print(f"        below refers to them.")
        print(f"    duplicate DisMuon tracks removed:                  {out['n_dup_removed']}")
        print(f"    events with 1 / 2 / 3+ DisMuons:                   "
              f"{out['n_mu1']} / {out['n_mu2']} / {out['n_mu3p']}")

        print()
        print("  STEP 2 -- FINDING THE TRUE MUON among the DisMuons of each event")
        print(f"    events where the true muon was found:              {n_gm}   ({pct(n_gm, D)} of selected)")
        print(f"    events where NO DisMuon matches the gen muon:      {out['n_no_gm']}   ({pct(out['n_no_gm'], D)})")
        print(f"        (signal muon not reconstructed, or only fake muons present)")
        print(f"    events with 2+ DisMuons inside the match cone:     {out['n_gm_multi']}")
        print(f"        (the same physical muon reconstructed more than once)")
        print(f"    cross-check -- true muon also flagged by NanoAOD")
        print(f"    (genPartFlav != 0):                                {out['n_gm_flavok']} / {n_gm}")
        print()
        print(f"    -- missing MB2 values (muon cannot be matched at station 2) --")
        print(f"    GenMuons with no usable eta/phi_at_mb2:            {out['n_gen_mb2_missing']} / {D} events"
              f"   ({pct(out['n_gen_mb2_missing'], D)})")
        print(f"    DisMuons with no usable eta/phi_at_mb2:            {out['n_mu_mb2_missing']} / {out['n_mu_total']} muons"
              f"   ({pct(out['n_mu_mb2_missing'], out['n_mu_total'])})")
        print(f"    events where EVERY DisMuon misses MB2 values:      {out['n_event_all_mu_mb2_missing']}")
        print(f"        (these events pile up in the last bin of the min-dR plot;")
        print(f"         see the *_mb2_missing_* plots for eta/phi/pt of the muons")
        print(f"         with vs without MB2 values)")
        print()
        print(f"    Is the true muon the highest-pT DisMuon of the event?")
        print(f"      yes:                                             {out['n_gm_lead']}   ({pct(out['n_gm_lead'], n_gm)} of found)")
        print(f"      no (a different muon has higher pT):             {out['n_gm_sub']}   ({pct(out['n_gm_sub'], n_gm)})")

        print()
        print("  STEP 3 -- QUALITY CUTS: pt>30 GeV, |eta|<2.4, mediumId,")
        print("            iso<0.18, 0.1<|dxy|<10 cm   (applied to DisMuons only)")
        print(f"    events where the true muon passes ALL cuts:        {out['n_oracle']}   ({pct(out['n_oracle'], n_gm)} of found)")
        print(f"    --> this is the BEST any selection method can do: if the true")
        print(f"        muon fails the cuts, no method can select it.")
        print(f"    events where 2+ muons pass all cuts:               {out['n_2pluspass']}")
        print(f"        (if 0, the choice of 'which passing muon to take' is")
        print(f"         always unambiguous)")

        print()
        print("  STEP 4 -- METHOD B (current): take the highest-pT DisMuon;")
        print("            keep the event only if THAT muon passes all cuts")
        print(f"    events KEPT by method B:                           {out['nB_cand']}   ({pct(out['nB_cand'], D)} of selected)")
        print(f"       the selected muon is the true muon:             {out['nB_correct']}")
        print(f"       the true muon is a different muon (mistake):    {out['nB_wrong']}")
        print(f"       the true muon was never found in the event:     {out['nB_nomatch']}")
        print(f"    events REJECTED by method B (highest-pT muon fails a cut):")
        print(f"       true muon found and PASSES all cuts:            {out['nB_lost_recov']}")
        print(f"          --> AVOIDABLE loss: the true muon is fine, it just is")
        print(f"              not the highest-pT muon; method C keeps these events")
        print(f"       true muon found but FAILS the cuts:             {out['nB_lost_unrec']}")
        print(f"          (lost with any method; in {out['nB_faillead_leadgm']} of these the failing")
        print(f"           highest-pT muon WAS the true muon itself)")
        print(f"       true muon was never found in the event:         {out['nB_nocand_nogm']}")

        print()
        print("  STEP 5 -- METHOD A/C: take the highest-pT DisMuon that PASSES")
        print("            all cuts (A deletes the failing muons afterwards,")
        print("            C keeps them; both select the SAME muon)")
        print(f"    events KEPT by method A/C:                         {out['nAC_cand']}   ({pct(out['nAC_cand'], D)} of selected)")
        print(f"       the selected muon is the true muon:             {out['nAC_correct']}")
        print(f"       true muon passes too but a higher-pT muon was")
        print(f"       selected instead (mistake):                     {out['nAC_wrong_outrank']}")
        print(f"       true muon fails cuts, a different muon was")
        print(f"       selected (mistake):                             {out['nAC_wrong_gmfail']}")
        print(f"       the true muon was never found in the event:     {out['nAC_nomatch']}")
        print(f"    events REJECTED by method A/C (no muon passes all cuts):")
        print(f"       true muon found but fails the cuts:             {out['nAC_lost_gm']}")
        print(f"       true muon was never found in the event:         {out['nAC_nocand_nogm']}")
        print()
        gain = out["nAC_correct"] - out["nB_correct"]
        print(f"    SUMMARY: method A/C selects the true muon in {gain} more")
        print(f"    events than method B ({out['nAC_correct']} vs {out['nB_correct']}).")

        print()
        print("  STEP 6 -- WHY KEEP THE FAILING MUONS (method C) INSTEAD OF")
        print("            DELETING THEM (method A)")
        print(f"    events where a muon deleted by method A is back-to-back")
        print(f"    (cos(angle) < {COSA_B2B}) with the selected signal muon:  {out['nA_prune_b2b']}")
        print(f"    Method A deletes these muons before the cosmic-ray veto runs,")
        print(f"    so the veto cannot use them.  On cosmic/NoBPTX data such a")
        print(f"    muon can be the second leg of a cosmic ray crossing the")
        print(f"    detector; method C keeps it available for the veto.")

        print()
        print("  STEP 7 -- WHICH CUT REJECTS THE MUONS")
        print(f"    'fails'      = the muon fails this cut (it may also fail others)")
        print(f"    'only this'  = the muon passes the other five cuts and fails")
        print(f"                   ONLY this one (what relaxing one cut would recover)")
        print(f"    highest-pT muon: one entry per selected event   ({D} events)")
        print(f"    true muon:       one entry per event where it was found   ({n_gm} events)")
        print()
        print(f"    {'':14}{'-- highest-pT muon --':>26}{'-- true muon --':>26}")
        print(f"    {'cut':<14}{'fails':>12}{'only this':>12}{'fails':>14}{'only this':>12}")
        for k, lab in CUT_DEFS:
            print(f"    {lab:<14}"
                  f"{out[f'cf_lead_{k}']:>6} {pct(out[f'cf_lead_{k}'], D):>7}"
                  f"{out[f'co_lead_{k}']:>10}"
                  f"{out[f'cf_gm_{k}']:>8} {pct(out[f'cf_gm_{k}'], n_gm):>7}"
                  f"{out[f'co_gm_{k}']:>10}")
        print("=" * 76 + "\n")

        # ── plots ──
        plot_mindr(out["h_mindr"],
                   os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_mindr.pdf"),
                   "Closest DisMuon to the GenMuon: dR at MB2 vs at vertex " + title_suffix)

        plot_simple(out["h_gm_rank"],
                    os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_gm_rank.pdf"),
                    "pT rank of the true muon among DisMuons (0 = highest pT) " + title_suffix,
                    color="darkblue")

        plot_overlay(out["h_nmu"],
                     os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_multiplicity.pdf"),
                     "DisMuons per event: all vs passing all quality cuts " + title_suffix)

        plot_overlay(out["h_gmpt_by_rank"],
                     os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_gmpt_by_rank.pdf"),
                     "True muon pT: is it the highest-pT DisMuon or not " + title_suffix,
                     normalize=True)

        # eta / phi / pt of muons with vs without usable MB2 values
        for coll, cname in (("reco", "DisMuon"), ("gen", "GenMuon")):
            for var in ("eta", "phi", "pt"):
                plot_overlay(
                    out[f"h_mb2_{var}_{coll}"],
                    os.path.join(OUTPUT_DIR,
                                 f"{PREFIX}{basename}_mb2_missing_{coll}_{var}.pdf"),
                    f"{cname} {var}: with vs without MB2 values " + title_suffix,
                    normalize=True)

        plot_pt2d(out["h_pt2d_lead_gm"],
                  os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_pt2d_lead_vs_gm.pdf"),
                  "Highest-pT DisMuon vs true muon pT (true muon not highest-pT) " + title_suffix)

        plot_cut_bars(out,
                      os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_cut_breakdown.pdf"),
                      "Fraction of muons failing each quality cut " + title_suffix)

        # method comparison: denominator = events where the true muon was found
        for var, xlabel in (("pt",  r"Gen muon $p_T$ [GeV]"),
                            ("lxy", r"Gen muon production $L_{xy}$ [cm]")):
            plot_multi_eff(
                out[f"h_{var}_denom"],
                [(out[f"h_{var}_oracle"], "best possible: true muon passes cuts", "gray"),
                 (out[f"h_{var}_AC"],     "method A/C: highest-pT passing muon",  "red"),
                 (out[f"h_{var}_B"],      "method B: highest-pT muon, then cuts", "blue")],
                xlabel,
                os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_eff_{var}.pdf"),
                "Fraction of events where the true muon is selected " + title_suffix,
                denom_label="events where the true muon was found")

            # how often the true muon is found at all, vs the same variable
            plot_multi_eff(
                out[f"h_{var}_evdenom"],
                [(out[f"h_{var}_denom"], "true muon found among DisMuons", "green")],
                xlabel,
                os.path.join(OUTPUT_DIR, f"{PREFIX}{basename}_matcheff_{var}.pdf"),
                "Fraction of events where the true muon is found " + title_suffix,
                denom_label="all selected events")

    print("\nAll samples done!")