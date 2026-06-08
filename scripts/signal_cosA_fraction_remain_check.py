import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import coffea.nanoevents.methods.vector as vector

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"


def plot_cosA_efficiency(h_all, h_post, OUTPUT_DIR, PREFIX, title_suffix, filename_suffix):
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

    ax_dist.step(edges[:-1], all_vals,  where="post", label="All Signal",       color="blue",  linestyle="--")
    ax_dist.step(edges[:-1], post_vals, where="post", label="Post-cosA Signal", color="red",   linestyle="-")
    ax_dist.set_ylabel("Events")
    ax_dist.set_title(r"$\cos\alpha < -0.99$ Veto Efficiency " + title_suffix)
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

    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}cosA_efficiency_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved efficiency plot: {outpath}")

def plot_timeNDof(h, OUTPUT_DIR, PREFIX, title_suffix, filename_suffix):
    if np.sum(h.values()) == 0:
        print("Skipping timeNDof plot (empty)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    vals  = h.values()
    edges = h.axes["val"].edges
    ax.step(edges[:-1], vals, where="post", color="purple")
    ax.set_xlabel("DisMuon timeNDof")
    ax.set_ylabel("DisMuons")
    ax.set_title("DisMuon timeNDof " + title_suffix)
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}timeNDof_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved timeNDof plot: {outpath}")


class CosASignalCheckProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            "n_events_initial": 0,
            "n_signal_denom": 0,
            "n_signal_numer": 0,
            "n_signal_numer_cosA_dt": 0,
            "n_events_2plus_muons": 0,
            "n_events_vetoed_cosA": 0,
            "n_events_vetoed_dt": 0,
            # ── Duplicate-removal counters ──
            "n_events_multi_dimuon_predup": 0,
            "n_events_multi_dimuon_postdup": 0,
            # Total duplicate tracks removed (before any cosA / dt cut)
            "n_duplicate_tracks_removed": 0,
            "n_signal_numer_dd": 0,
            "n_signal_numer_cosA_dt_dd": 0,
            "n_events_2plus_muons_dd": 0,
            "n_events_vetoed_cosA_dd": 0,
            "n_events_vetoed_dt_dd": 0,
            "lead_pt_all": Hist(
                axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")
            ),
            # Numerator: leading DisMuon pT after cosA veto only
            "lead_pt_post_cosA": Hist(
                axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")
            ),
            # Numerator: leading DisMuon pT after cosA + dt veto
            "lead_pt_post_cosA_dt": Hist(
                axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")
            ),
            # Numerator (after duplicate removal): leading DisMuon pT after cosA veto only
            "lead_pt_post_cosA_dd": Hist(
                axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")
            ),
            # Numerator (after duplicate removal): leading DisMuon pT after cosA + dt veto
            "lead_pt_post_cosA_dt_dd": Hist(
                axis.Regular(50, 0, 500, name="val", label=r"Leading DisMuon $p_T$ [GeV]")
            ),
            "timeNDof_all": Hist(
                axis.Regular(50, 0, 50, name="val", label="DisMuon timeNDof")
            ),
        }

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")
        self.output["n_events_initial"] += len(events)

        is_signal = not ("Cosmic" in dataset or dataset.startswith("LooseMu")
                         or dataset == "test_cosmics_calib" or "NoBPTX" in dataset)
        has_gen = "GenPart" in events.fields

        if not (is_signal and has_gen):
            return self.output

        # ── Rebuild DisMuon as Lorentz vectors ──
        events["DisMuon"] = ak.zip(
            {
                "pt":                       events.DisMuon.pt,
                "eta":                      events.DisMuon.eta,
                "phi":                      events.DisMuon.phi,
                "mass":                     events.DisMuon.mass,
                "charge":                   events.DisMuon.charge,
                "timeNDof":                 events.DisMuon.timeNDof,
                "isStandalone":             events.DisMuon.isStandalone,
                "isGlobal":                 events.DisMuon.isGlobal,
                "dxy":                      events.DisMuon.dxy,
                "dz":                       events.DisMuon.dz,
                "numberOfValidMuonDTHits":  events.DisMuon.numberOfValidMuonDTHits,
                "numberOfValidMuonCSCHits": events.DisMuon.numberOfValidMuonCSCHits,
                "numberOfValidMuonHits":    events.DisMuon.numberOfValidMuonHits,
                "dtStationsWithValidHits":  events.DisMuon.dtStationsWithValidHits,
                "timeAtIpInOut":            events.DisMuon.timeAtIpInOut,
                "timeAtIpInOutErr":         events.DisMuon.timeAtIpInOutErr,
                "mediumId":                 events.DisMuon.mediumId,
                "pfRelIso03_all":           events.DisMuon.pfRelIso03_all,
                "eta_at_mb2":               events.DisMuon.eta_at_mb2,
                "phi_at_mb2":               events.DisMuon.phi_at_mb2,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # ── GenPart ──
        events["GenPart"] = ak.zip(
            {
                "pt":         events.GenPart.pt,
                "eta":        events.GenPart.eta,
                "phi":        events.GenPart.phi,
                "mass":       events.GenPart.mass,
                "pdgId":      events.GenPart.pdgId,
                "status":     events.GenPart.status,
                "eta_at_mb2": events.GenPart.eta_at_mb2,
                "phi_at_mb2": events.GenPart.phi_at_mb2,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
        gen_muons = events.GenPart[
            (abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1)
        ]
        gen_muons = gen_muons[(gen_muons.pt > 30) & (abs(gen_muons.eta) < 2.4)]

        # ── Kinematic cuts on DisMuons ──
        dismuon_mask = (events.DisMuon.pt > 30) & (abs(events.DisMuon.eta) < 2.4)
        events["DisMuon"] = events.DisMuon[dismuon_mask]

        # ── Signal-only event filtering ──
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
        signal_mask = (ak.num(gen_muons) == 1) & (ak.num(jets) == 1) & good_MET

        events    = events[signal_mask]
        gen_muons = gen_muons[signal_mask]

        # ── Require at least 1 DisMuon ──
        mask_has_muons = ak.num(events.DisMuon) > 0
        if ak.sum(mask_has_muons) == 0:
            return self.output

        events    = events[mask_has_muons]
        gen_muons = gen_muons[mask_has_muons]

        # ── Sort by pT, apply quality cuts to the leading muon only ──
        sorted_muons = events.DisMuon[
            ak.argsort(events.DisMuon.pt, axis=1, ascending=False)
        ]
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

        lead = sorted_muons[:, 0]
        n_muons = ak.num(sorted_muons)

        # ══════════════════════════════════════════
        # DENOMINATOR: all signal events passing cuts
        # (shared between the with- and without-duplicate-removal
        #  paths — removing duplicate tracks never drops whole events)
        # ══════════════════════════════════════════
        self.output["n_signal_denom"] += len(sorted_muons)
        self.output["lead_pt_all"].fill(val=lead.pt)
        self.output["timeNDof_all"].fill(val=ak.to_numpy(ak.flatten(sorted_muons.timeNDof)))

        # ╔══════════════════════════════════════════════════════════╗
        # ║  DUPLICATE TRACK REMOVAL — applied BEFORE the cosA / dt   ║
        # ║  vetoes, so we can see how many duplicate tracks are      ║
        # ║  removed prior to those selections.  A muon is removed    ║
        # ║  as a duplicate if it matches an earlier (higher-pT),     ║
        # ║  same-charge muon in (eta, phi, pT); the higher-pT one    ║
        # ║  is kept.  The cosA / dt vetoes below are then run twice  ║
        # ║  — once on the full collection (sorted_muons) and once    ║
        # ║  on the duplicate-removed collection (sorted_muons_dd) —  ║
        # ║  so signal efficiency can be compared with vs. without    ║
        # ║  removing duplicates first.  The denominator is unchanged ║
        # ║  (removing duplicate tracks never drops whole events).    ║
        # ╚══════════════════════════════════════════════════════════╝
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
        is_duplicate = ak.any(match, axis=2)
        sorted_muons_dd = sorted_muons[~is_duplicate]

        self.output["n_events_multi_dimuon_predup"]  += int(ak.sum(ak.num(sorted_muons)    > 1))
        self.output["n_events_multi_dimuon_postdup"] += int(ak.sum(ak.num(sorted_muons_dd) > 1))
        # Total duplicate tracks removed at this stage (before any cosA / dt cut)
        self.output["n_duplicate_tracks_removed"] += int(
            ak.sum(ak.num(sorted_muons)) - ak.sum(ak.num(sorted_muons_dd))
        )

        # leading muon is never removed (nothing higher-pT precedes it)
        lead_dd    = sorted_muons_dd[:, 0]
        n_muons_dd = ak.num(sorted_muons_dd)

        # ══════════════════════════════════════════
        # cosA VETO: for events with 2+ DisMuons,
        # compute cosA between the leading muon and
        # every subsequent muon.  If ANY pair has
        # cosA < -0.99, veto the entire event.
        # ══════════════════════════════════════════
        has_2plus = (n_muons >= 2)
        self.output["n_events_2plus_muons"] += int(ak.sum(has_2plus))

        survives_cosA = np.ones(len(sorted_muons), dtype=bool)

        if ak.sum(has_2plus) > 0:
            multi_muons = sorted_muons[has_2plus]
            multi_lead  = multi_muons[:, 0]
            sub_muons   = multi_muons[:, 1:]

            # cosA between lead and each sub-leading muon
            dot = (multi_lead.px * sub_muons.px +
                   multi_lead.py * sub_muons.py +
                   multi_lead.pz * sub_muons.pz)
            mag = multi_lead.p * sub_muons.p
            cosA = ak.where(mag > 0, dot / mag, -1000.0)

            # Veto if ANY sub-leading muon gives cosA < -0.99
            any_backtoback = ak.any(cosA < -0.99, axis=1)

            pair_idx = np.where(ak.to_numpy(has_2plus))[0]
            survives_cosA[pair_idx] = ~ak.to_numpy(any_backtoback)

        n_vetoed_cosA = int(np.sum(~survives_cosA))
        self.output["n_events_vetoed_cosA"] += n_vetoed_cosA

        # ══════════════════════════════════════════
        # NUMERATOR (cosA only): signal events surviving cosA veto
        # ══════════════════════════════════════════
        survives_cosA_ak = ak.Array(survives_cosA)
        self.output["n_signal_numer"] += int(ak.sum(survives_cosA_ak))
        self.output["lead_pt_post_cosA"].fill(val=lead.pt[survives_cosA_ak])

        # ══════════════════════════════════════════
        # dt VETO: for events surviving cosA, separate
        # into upper (phi > 0) and lower (phi < 0),
        # pick highest pT in each hemisphere,
        # compute dt = upper - lower.
        # If dt < -20 ns, veto the entire event.
        # ══════════════════════════════════════════
        survives_cosA_dt = survives_cosA.copy()

        # Only apply dt cut to events that survived cosA AND have 2+ muons
        cosA_surv_and_2plus = survives_cosA & ak.to_numpy(has_2plus)

        if np.sum(cosA_surv_and_2plus) > 0:
            dt_muons = sorted_muons[ak.Array(cosA_surv_and_2plus)]

            dt_upper_all = dt_muons[dt_muons.phi > 0]
            dt_lower_all = dt_muons[dt_muons.phi < 0]

            dt_has_both = (ak.num(dt_upper_all) >= 1) & (ak.num(dt_lower_all) >= 1)

            if ak.sum(dt_has_both) > 0:
                dt_upper_both = dt_upper_all[dt_has_both]
                dt_lower_both = dt_lower_all[dt_has_both]

                # Pick highest pT in each hemisphere
                dt_upper_sorted = dt_upper_both[ak.argsort(dt_upper_both.pt, axis=1, ascending=False)]
                dt_lower_sorted = dt_lower_both[ak.argsort(dt_lower_both.pt, axis=1, ascending=False)]

                dt_upper = dt_upper_sorted[:, 0]
                dt_lower = dt_lower_sorted[:, 0]

                dt_val = dt_upper.timeAtIpInOut - dt_lower.timeAtIpInOut
                dt_fail = ak.to_numpy(dt_val < -20)

                # Map back to the full array
                cosA_surv_2plus_idx = np.where(cosA_surv_and_2plus)[0]
                has_both_idx = cosA_surv_2plus_idx[ak.to_numpy(dt_has_both)]
                fail_idx = has_both_idx[dt_fail]

                survives_cosA_dt[fail_idx] = False

        n_vetoed_dt = int(np.sum(survives_cosA & ~survives_cosA_dt))
        self.output["n_events_vetoed_dt"] += n_vetoed_dt

        # ══════════════════════════════════════════
        # NUMERATOR (cosA + dt): signal events surviving both vetoes
        # ══════════════════════════════════════════
        survives_cosA_dt_ak = ak.Array(survives_cosA_dt)
        self.output["n_signal_numer_cosA_dt"] += int(ak.sum(survives_cosA_dt_ak))
        self.output["lead_pt_post_cosA_dt"].fill(val=lead.pt[survives_cosA_dt_ak])

        # ══════════════════════════════════════════════════════════
        # REPEAT cosA / dt VETOES on the duplicate-removed collection
        # (sorted_muons_dd was built above, before any cosA / dt cut),
        # so signal efficiency can be compared with vs. without
        # removing duplicate tracks first.
        # ══════════════════════════════════════════════════════════

        # ── cosA VETO (after duplicate removal) ──
        has_2plus_dd = (n_muons_dd >= 2)
        self.output["n_events_2plus_muons_dd"] += int(ak.sum(has_2plus_dd))

        survives_cosA_dd = np.ones(len(sorted_muons_dd), dtype=bool)

        if ak.sum(has_2plus_dd) > 0:
            multi_muons_dd = sorted_muons_dd[has_2plus_dd]
            multi_lead_dd  = multi_muons_dd[:, 0]
            sub_muons_dd   = multi_muons_dd[:, 1:]

            dot_dd = (multi_lead_dd.px * sub_muons_dd.px +
                      multi_lead_dd.py * sub_muons_dd.py +
                      multi_lead_dd.pz * sub_muons_dd.pz)
            mag_dd = multi_lead_dd.p * sub_muons_dd.p
            cosA_dd = ak.where(mag_dd > 0, dot_dd / mag_dd, -1000.0)

            any_backtoback_dd = ak.any(cosA_dd < -0.99, axis=1)

            pair_idx_dd = np.where(ak.to_numpy(has_2plus_dd))[0]
            survives_cosA_dd[pair_idx_dd] = ~ak.to_numpy(any_backtoback_dd)

        n_vetoed_cosA_dd = int(np.sum(~survives_cosA_dd))
        self.output["n_events_vetoed_cosA_dd"] += n_vetoed_cosA_dd

        # ── NUMERATOR (cosA only, after duplicate removal) ──
        survives_cosA_dd_ak = ak.Array(survives_cosA_dd)
        self.output["n_signal_numer_dd"] += int(ak.sum(survives_cosA_dd_ak))
        self.output["lead_pt_post_cosA_dd"].fill(val=lead_dd.pt[survives_cosA_dd_ak])

        # ── dt VETO (after duplicate removal) ──
        survives_cosA_dt_dd = survives_cosA_dd.copy()

        cosA_surv_and_2plus_dd = survives_cosA_dd & ak.to_numpy(has_2plus_dd)

        if np.sum(cosA_surv_and_2plus_dd) > 0:
            dt_muons_dd = sorted_muons_dd[ak.Array(cosA_surv_and_2plus_dd)]

            dt_upper_all_dd = dt_muons_dd[dt_muons_dd.phi > 0]
            dt_lower_all_dd = dt_muons_dd[dt_muons_dd.phi < 0]

            dt_has_both_dd = (ak.num(dt_upper_all_dd) >= 1) & (ak.num(dt_lower_all_dd) >= 1)

            if ak.sum(dt_has_both_dd) > 0:
                dt_upper_both_dd = dt_upper_all_dd[dt_has_both_dd]
                dt_lower_both_dd = dt_lower_all_dd[dt_has_both_dd]

                dt_upper_sorted_dd = dt_upper_both_dd[ak.argsort(dt_upper_both_dd.pt, axis=1, ascending=False)]
                dt_lower_sorted_dd = dt_lower_both_dd[ak.argsort(dt_lower_both_dd.pt, axis=1, ascending=False)]

                dt_upper_dd = dt_upper_sorted_dd[:, 0]
                dt_lower_dd = dt_lower_sorted_dd[:, 0]

                dt_val_dd = dt_upper_dd.timeAtIpInOut - dt_lower_dd.timeAtIpInOut
                dt_fail_dd = ak.to_numpy(dt_val_dd < -20)

                cosA_surv_2plus_idx_dd = np.where(cosA_surv_and_2plus_dd)[0]
                has_both_idx_dd = cosA_surv_2plus_idx_dd[ak.to_numpy(dt_has_both_dd)]
                fail_idx_dd = has_both_idx_dd[dt_fail_dd]

                survives_cosA_dt_dd[fail_idx_dd] = False

        n_vetoed_dt_dd = int(np.sum(survives_cosA_dd & ~survives_cosA_dt_dd))
        self.output["n_events_vetoed_dt_dd"] += n_vetoed_dt_dd

        # ── NUMERATOR (cosA + dt, after duplicate removal) ──
        survives_cosA_dt_dd_ak = ak.Array(survives_cosA_dt_dd)
        self.output["n_signal_numer_cosA_dt_dd"] += int(ak.sum(survives_cosA_dt_dd_ak))
        self.output["lead_pt_post_cosA_dt_dd"].fill(val=lead_dd.pt[survives_cosA_dt_dd_ak])

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

        denom       = out["n_signal_denom"]
        numer_cosA  = out["n_signal_numer"]
        numer_both  = out["n_signal_numer_cosA_dt"]
        n2plus      = out["n_events_2plus_muons"]
        vetoed_cosA = out["n_events_vetoed_cosA"]
        vetoed_dt   = out["n_events_vetoed_dt"]

        numer_cosA_dd  = out["n_signal_numer_dd"]
        numer_both_dd  = out["n_signal_numer_cosA_dt_dd"]
        n2plus_dd      = out["n_events_2plus_muons_dd"]
        vetoed_cosA_dd = out["n_events_vetoed_cosA_dd"]
        vetoed_dt_dd   = out["n_events_vetoed_dt_dd"]
        dup_removed    = out["n_duplicate_tracks_removed"]

        def eff(n):
            return f"{n / denom * 100:.4f}%" if denom > 0 else "n/a"

        print("\n" + "=" * 60)
        print(f"  {basename}   (signal events: {denom})")
        print("=" * 60)
        print("  Duplicate removal (before cosA / dt cuts):")
        print(f"    Duplicate tracks removed:  {dup_removed}")
        print(f"    Events with 2+ DisMuons:   {n2plus} -> {n2plus_dd}")
        print()
        print(f"    {'':25}{'without dup.':>13}{'with dup.':>13}")
        print(f"    {'Vetoed by cosA':25}{vetoed_cosA:>13}{vetoed_cosA_dd:>13}")
        print(f"    {'Vetoed by dt':25}{vetoed_dt:>13}{vetoed_dt_dd:>13}")
        print(f"    {'cosA efficiency':25}{eff(numer_cosA):>13}{eff(numer_cosA_dd):>13}")
        print(f"    {'cosA + dt efficiency':25}{eff(numer_both):>13}{eff(numer_both_dd):>13}")
        print()
        print(f"    Signal events saved from veto by removing duplicates  —  "
              f"cosA: {numer_cosA_dd - numer_cosA}   cosA + dt: {numer_both_dd - numer_both}")
        print("=" * 60 + "\n")

        # Plot cosA-only efficiency
        plot_cosA_efficiency(
            out["lead_pt_all"],
            out["lead_pt_post_cosA"],
            OUTPUT_DIR, PREFIX,
            title_suffix=title_suffix,
            filename_suffix=basename,
        )

        # Plot cosA + dt combined efficiency
        plot_cosA_efficiency(
            out["lead_pt_all"],
            out["lead_pt_post_cosA_dt"],
            OUTPUT_DIR, PREFIX,
            title_suffix=title_suffix + r" (cosA + $\Delta t$)",
            filename_suffix=basename + "_cosA_dt",
        )

        # Plot cosA-only efficiency (after duplicate removal)
        plot_cosA_efficiency(
            out["lead_pt_all"],
            out["lead_pt_post_cosA_dd"],
            OUTPUT_DIR, PREFIX,
            title_suffix=title_suffix + " (after duplicate removal)",
            filename_suffix=basename + "_dupremoved",
        )

        # Plot cosA + dt combined efficiency (after duplicate removal)
        plot_cosA_efficiency(
            out["lead_pt_all"],
            out["lead_pt_post_cosA_dt_dd"],
            OUTPUT_DIR, PREFIX,
            title_suffix=title_suffix + r" (after duplicate removal, cosA + $\Delta t$)",
            filename_suffix=basename + "_dupremoved_cosA_dt",
        )

        plot_timeNDof(
            out["timeNDof_all"],
            OUTPUT_DIR, PREFIX,
            title_suffix=title_suffix,
            filename_suffix=basename,
        )

    print("\nAll samples done!")