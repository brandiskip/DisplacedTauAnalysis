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


def get_Lxy(genvistau):
    """Transverse decay length: tau production vertex (= stau decay vertex)
    minus the stau production vertex (= PV)."""
    vx = genvistau.parent.vx - genvistau.parent.distinctParent.vx
    vy = genvistau.parent.vy - genvistau.parent.distinctParent.vy
    return np.sqrt(vx ** 2 + vy ** 2)


def _pmvec(pt, eta, phi, mass):
    """Momentum 4-vector via the PtEtaPhiMLorentzVector behavior."""
    return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass},
                  with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)


def _cosA_vec(u, v):
    """cos(opening angle) between two momentum vectors."""
    denom = u.p * v.p
    return ak.where(denom != 0, (u.px * v.px + u.py * v.py + u.pz * v.pz) / denom, np.nan)


def save_muon_type_counts(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix=""):
    """Grouped bar chart of Standalone / Global / Tracker DisMuon counts.
    Categories OVERLAP — a muon is counted once in each bin it satisfies."""
    if np.sum(h.values()) == 0:
        print(f"Skipping {var_name} (Empty)")
        return

    labels = ["Standalone", "Global", "Tracker"]
    cats = list(h.axes["cat"])
    x = np.arange(len(labels))
    width = 0.8 / max(len(cats), 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cat in enumerate(cats):
        counts = h[{"cat": cat}].values()
        offset = (i - (len(cats) - 1) / 2.0) * width
        bars = ax.bar(x + offset, counts, width=width, label=cat)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(c)}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("DisMuons")
    ax.set_xlabel("Muon type (categories overlap)")
    ax.set_title(f"DisMuon types {title_suffix}")
    ax.legend(title="Dataset")

    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{filename_suffix}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved muon-type plot to: {outpath}")


def save_comparison_overlay(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix="", log_y=False, normalize=True):
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
            (h_slice * (1.0 / total_events)).plot1d(ax=ax, label=label)
        else:
            h_slice.plot1d(ax=ax, label=label)

    ax.legend(title="Category")
    ax.set_ylabel("Fraction of Events" if normalize else "Events")
    ax.set_title(f"Comparison: {var_name} {title_suffix}")
    if log_y:
        ax.set_yscale("log")

    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{filename_suffix}.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved comparison plot to: {outpath}")

def save_time_overlay(h, sel, PREFIX, OUTPUT_DIR, title_suffix="", normalize=True):
    if sel not in list(h.axes["sel"]):
        print(f"Skipping time plot {sel} (category never filled)")
        return
    h_sel = h[{"sel": sel}]
    if np.sum(h_sel.values()) == 0:
        print(f"Skipping time plot {sel} (Empty)")
        return

    edges = h_sel.axes["val"].edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_window = (centers > -45) & (centers < 20)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label in h_sel.axes["cat"]:
        vals = h_sel[{"cat": label}].values()
        tot = vals.sum()
        if tot == 0:
            continue
        frac_in = vals[in_window].sum() / tot
        if normalize:
            vals = vals / tot
        ax.step(edges[:-1], vals, where="post",
                label=f"{label} (n={int(tot)}, {100*frac_in:.1f}% in window)")

    ax.set_yscale("log")
    for t in (-50, -25, 25, 50):
        ax.axvline(t, color="gray", ls=":", alpha=0.6)
    ax.axvspan(-45, 20, color="green", alpha=0.10, label="inTimeMuon keeps (~ -45 to +20 ns)")
    ax.axvline(-45, color="green", ls="--", alpha=0.8)
    ax.axvline(+20, color="green", ls="--", alpha=0.8)
    ax.set_xlabel(h_sel.axes["val"].label)
    ax.set_ylabel("Fraction of DisMuons / (2 ns)" if normalize else "DisMuons / (2 ns)")
    ax.set_title(f"timeAtIpInOut — {sel} {title_suffix}")
    ax.legend()

    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}time_{sel}.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved time plot: {outpath}")

class SingleMuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            "n_events_initial": 0,
            "n_events_total_cosmic": 0,
            "n_events_total_nobptx": 0,
            "n_evt_single_mu_raw_cosmic": 0,
            "n_evt_single_mu_raw_nobptx": 0,
            # NoBPTX events containing a candidate-based duplicate (INFO ONLY — no longer vetoed;
            # duplicates now go through the outertrack/segment cutflow like cosmic MC)
            "n_events_with_dup_nobptx": 0,

            # ── duplicate-pair cutflow (cosmic MC + NoBPTX v21 data) ──
            "n_dup_pairs_total_cosmic": 0,
            "n_dup_removed_outer_cosmic": 0,
            "n_dup_removed_seg_cosmic": 0,
            "n_dup_pairs_surviving_cosmic": 0,
            "n_dup_pairs_total_nobptx": 0,
            "n_dup_removed_outer_nobptx": 0,
            "n_dup_removed_seg_nobptx": 0,
            "n_dup_pairs_surviving_nobptx": 0,

            # ── per-pair dump: duplicate pairs that SURVIVE the outertrack + segment
            #    cosA<-0.8 cuts (still flagged as duplicates). Written to a .txt at the
            #    end so we can inspect run/lumi/event + kinematics and see why they
            #    still look like duplicates. One row per surviving duplicate pair. ──
            "dup_surv_rows_cosmic": [],
            "dup_surv_rows_nobptx": [],

            # ── single-muon events: pass method C candidate cuts but have exactly 1
            #    DisMuon, so they never enter the dup/cosA/dt cutflow ──
            "n_evt_single_mu_cosmic": 0,
            "n_evt_single_mu_nobptx": 0,

            # ── event-level cosmic-removal cutflow ──
            "n_evt_enter_cosmic": 0, "n_evt_after_dup_cosmic": 0, "n_evt_after_cosA_cosmic": 0, "n_evt_after_dt_cosmic": 0,
            "n_evt_enter_nobptx": 0, "n_evt_after_dup_nobptx": 0, "n_evt_after_cosA_nobptx": 0, "n_evt_after_dt_nobptx": 0,

            # ── inTimeMuon-on-non-candidate study: original >=2mu count, and how many of
            #    those collapse to candidate-only once out-of-time partners are dropped ──
            "n_evt_ge2_orig_cosmic": 0, "n_evt_itm_to_1mu_cosmic": 0,
            "n_evt_ge2_orig_nobptx": 0, "n_evt_itm_to_1mu_nobptx": 0,

            # ── Standalone / Global / Tracker muon counts (categories overlap) ──
            "muon_type": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True),
                axis.IntCategory([0, 1, 2], name="type", label="Muon Type"),
            ),

            # ── uncut upper vs lower (exactly-two-muon events) ──
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

            # ── cosA cutflow histograms ──
            "cosA_study_lead_vs_sub": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(200, -1.01, 1.0, name="val", label=r"$\cos\alpha$ (Candidate vs Each Retained Muon)")
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
            "time_at_ip": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True),
                axis.StrCategory([], name="sel", label="Selection", growth=True),
                axis.Regular(300, -300, 300, name="val", label=r"DisMuon timeAtIpInOut [ns]"),
            ),
        }

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")
        self.output["n_events_initial"] += len(events)

        # ── Dataset classification (up top so cosmic-only branches can be added conditionally) ──
        is_cosmic = "Cosmic" in dataset or dataset.startswith("LooseMu") or dataset == "test_cosmics_calib"
        is_nobptx = "NoBPTX" in dataset
        is_signal = not (is_cosmic or is_nobptx)
        ds_label = "MC" if is_cosmic else ("Data" if is_nobptx else "Signal")
        has_gen = "GenPart" in events.fields

        if is_signal:
            events["DisMuon"] = events.DisMuon[events.DisMuon.inTimeMuon == True]

        if is_cosmic:
            self.output["n_events_total_cosmic"] += len(events)
            self.output["n_evt_single_mu_raw_cosmic"] += int(ak.sum(ak.num(events.DisMuon) == 1))
        elif is_nobptx:
            self.output["n_events_total_nobptx"] += len(events)
            self.output["n_evt_single_mu_raw_nobptx"] += int(ak.sum(ak.num(events.DisMuon) == 1))

        # ── Base DisMuon fields; outertrack + segment exist for COSMIC MC and NoBPTX v21 ──
        dismuon_fields = {
            "pt": events.DisMuon.pt,
            "eta": events.DisMuon.eta,
            "phi": events.DisMuon.phi,
            "mass": events.DisMuon.mass,
            "charge": events.DisMuon.charge,
            "timeNDof": events.DisMuon.timeNDof,
            "isStandalone": events.DisMuon.isStandalone,
            "isGlobal": events.DisMuon.isGlobal,
            "isTracker": events.DisMuon.isTracker,
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
            "inTimeMuon": events.DisMuon.inTimeMuon,   # carried so we can require it on non-candidates
        }
        if is_cosmic or is_nobptx:
            dismuon_fields["outertrack_pt"]  = events.DisMuon.outertrack_pt
            dismuon_fields["outertrack_eta"] = events.DisMuon.outertrack_eta
            dismuon_fields["outertrack_phi"] = events.DisMuon.outertrack_phi
            dismuon_fields["sumSegX"] = events.DisMuon.sumSegX
            dismuon_fields["sumSegY"] = events.DisMuon.sumSegY
            dismuon_fields["sumSegZ"] = events.DisMuon.sumSegZ
            dismuon_fields["nSeg"]    = events.DisMuon.nSeg

        events["DisMuon"] = ak.zip(
            dismuon_fields,
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # ════════════════════════════════════════════════════════════
        # GEN-LEVEL SIGNAL REGION (signal samples only) — applied immediately
        # ════════════════════════════════════════════════════════════
        if is_signal and has_gen:
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

            signal_gen_mask = (
                (ak.num(events.GenVisStauTaus) == 1) &
                (ak.num(events.GenMuon) == 1) &
                (ak.num(events.GenElectron) == 0)
            )
            events = events[signal_gen_mask]

        # ══════════════════════════════════════════════════════════════════
        # METHOD C CANDIDATE SELECTION (all datasets)
        #   Every muon is tested against ALL quality cuts per-muon:
        #       pt > 30, |eta| < 2.4, mediumId, pfRelIso03_all < 0.18
        #       (signal only: 0.1 < |dxy| < 10 cm as well).
        #   The CANDIDATE is the highest-pT muon that passes — it need NOT be
        #   the highest-pT muon of the event. Every other muon is RETAINED
        #   with no cuts. The muon list is reordered so the CANDIDATE IS
        #   ALWAYS INDEX 0 (partners follow in pT order); events without a
        #   candidate get an empty DisMuon list and are dropped downstream.
        # ══════════════════════════════════════════════════════════════════
        sorted_all_kin = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        pass_all_kin = (
            (sorted_all_kin.pt > 30) &
            (abs(sorted_all_kin.eta) < 2.4) &
            (sorted_all_kin.mediumId == True) &
            (sorted_all_kin.pfRelIso03_all < 0.18)
        )
        if is_signal:
            pass_all_kin = pass_all_kin & (abs(sorted_all_kin.dxy) > 0.1) & (abs(sorted_all_kin.dxy) < 10)
        li_kin = ak.local_index(sorted_all_kin.pt, axis=1)
        cand_idx_kin = ak.firsts(li_kin[pass_all_kin])      # per-event index of the candidate; None if no muon passes
        has_cand_kin = ~ak.is_none(cand_idx_kin)
        # candidate-first reorder: [candidate] + the remaining muons in pT order.
        # ak.singletons -> [] for no-candidate events, so their order is the identity
        # (they are emptied by the keep mask right after anyway).
        order_kin = ak.concatenate(
            [ak.singletons(cand_idx_kin),
             li_kin[li_kin != ak.fill_none(cand_idx_kin, -1)]],
            axis=1)
        reordered_kin = sorted_all_kin[order_kin]
        keep_per_muon_kin, _ = ak.broadcast_arrays(has_cand_kin, reordered_kin.pt)
        events["DisMuon"] = reordered_kin[keep_per_muon_kin]

        dis_muons = events.DisMuon
        n_dismuons = ak.num(dis_muons)

        # ── NoBPTX: remove events with good vertices ──
        if is_nobptx:
            no_vertex_mask = (events.PV.npvsGood == 0)
            events = events[no_vertex_mask]
            dis_muons = events.DisMuon
            n_dismuons = ak.num(dis_muons)

        # ── Signal-only event filtering (jets + MET) ──
        if is_signal and has_gen:
            charged_sel = events.Jet.constituents.pf.charge != 0
            dxy = ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis=-1), -999,
                           ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis=-1))
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
            good_MET = (events.PFMET.pt > 105)
            signal_mask = (ak.num(jets) == 1) & good_MET
            events = events[signal_mask]
            dis_muons = events.DisMuon
            n_dismuons = ak.num(dis_muons)

        # ==========================================
        # GLOBAL DUPLICATE TRACK REMOVAL
        #   Cosmic MC + NoBPTX v21: duplicates are RETAINED in dis_muons_full and
        #   handled by the outertrack/segment cosA cutflow below; dis_muons is
        #   de-duplicated for ancillary studies only.
        #   Signal: whole-event veto on any duplicate (unchanged).
        # ==========================================
        dis_muons_full = dis_muons          # default (used if the block below is skipped)
        mask_has_muons_g = (n_dismuons >= 1)
        if ak.sum(mask_has_muons_g) > 0:
            events = events[mask_has_muons_g]
            dis_muons = dis_muons[mask_has_muons_g]

            # METHOD C: dis_muons is already CANDIDATE-FIRST (index 0 = candidate, which
            # passed pt/eta/mediumId/iso[/dxy] per-muon). Do NOT re-sort by pT here —
            # that would demote a candidate that is not the overall-highest-pT muon —
            # and no further lead-quality gate is needed: events without a candidate
            # were already dropped at the method C selection.
            sorted_all = dis_muons
            dis_muons_full = sorted_all      # cosmic + nobptx keep duplicates; overridden below for signal

            if len(events) > 0:
                lead_for_dup = sorted_all[:, 0]   # = the method C candidate
                deta_g = sorted_all.eta - lead_for_dup.eta
                dphi_g = sorted_all.delta_phi(lead_for_dup)
                mask_sc_g = (sorted_all.charge * lead_for_dup.charge) > 0

                is_duplicate_g = mask_sc_g & (abs(deta_g) < 0.01) & (abs(dphi_g) < 0.001)
                is_duplicate_g = is_duplicate_g & (ak.local_index(sorted_all, axis=1) > 0)

                if is_cosmic or is_nobptx:
                    # De-duplicate dis_muons for ancillary studies; keep the full
                    # (duplicate-retaining) dis_muons_full for the outertrack/segment cutflow.
                    if is_nobptx:
                        # info only: how many NoBPTX events contain a candidate-based duplicate
                        self.output["n_events_with_dup_nobptx"] += int(ak.sum(ak.any(is_duplicate_g, axis=1)))
                    dis_muons = sorted_all[~is_duplicate_g]
                else:
                    # Signal: VETO THE WHOLE EVENT if it has ANY duplicate.
                    has_dup = ak.any(is_duplicate_g, axis=1)
                    events = events[~has_dup]
                    sorted_all = sorted_all[~has_dup]
                    dis_muons = sorted_all
                    dis_muons_full = sorted_all

        n_dismuons = ak.num(dis_muons)

        # ── Standalone / Global / Tracker counts (categories overlap) ──
        if ak.sum(n_dismuons) > 0:
            self.output["muon_type"].fill(
                cat=ds_label,
                type=[0, 1, 2],
                weight=[
                    int(ak.sum(ak.flatten(dis_muons.isStandalone))),
                    int(ak.sum(ak.flatten(dis_muons.isGlobal))),
                    int(ak.sum(ak.flatten(dis_muons.isTracker))),
                ],
            )

        # ── Exactly-two-muon (uncut) upper vs lower ──
        mask_exactly_two_uncut = (n_dismuons == 2)
        if ak.sum(mask_exactly_two_uncut) > 0:
            two_muons_uncut = dis_muons[mask_exactly_two_uncut]
            sorted_by_phi = two_muons_uncut[ak.argsort(two_muons_uncut.phi, axis=1, ascending=False)]
            upper_candidates = sorted_by_phi[:, 0]
            lower_candidates = sorted_by_phi[:, 1]
            mask_opposite_hemispheres = (upper_candidates.phi > 0) & (lower_candidates.phi < 0)
            upper_final = upper_candidates[mask_opposite_hemispheres]
            lower_final = lower_candidates[mask_opposite_hemispheres]
            self.output["uncut_upper_vs_lower_phi"].fill(cat=ds_label, upper=upper_final.phi, lower=lower_final.phi)
            self.output["uncut_upper_vs_lower_eta"].fill(cat=ds_label, upper=upper_final.eta, lower=lower_final.eta)

        # ==========================================
        # COSMIC / NoBPTX cosmic-removal CUTFLOW  (>= 2 muons; all pairs considered)
        #   METHOD C: index 0 = the candidate (passed all quality cuts); all other
        #   muons are retained partners with no cuts.
        #   Pre-step: require inTimeMuon==True on every NON-CANDIDATE muon (candidate
        #       kept), applied BEFORE [1]/[2]/[3]. Events left with only the candidate
        #       survive.
        #   [1] duplicate removal (outertrack cosA < -0.8, then segment cosA < -0.8)
        #   [2] standard cosA < -0.99 (candidate vs each retained muon)
        #   [3] dt < -20 ns (ndof > 7 on the two upper/lower leaders ONLY)
        #   Single-muon events never enter this cutflow — counted separately.
        # ==========================================
        if is_cosmic or is_nobptx:
            prefix_cs = "cosmic" if is_cosmic else "nobptx"
            src = dis_muons_full   # duplicate-retaining, CANDIDATE-FIRST (method C applied)

            n_mu_src = ak.num(src)
            # single-muon events: untouched (that muon IS the candidate and already
            # passed pt/eta/mediumId/iso); no inTimeMuon requirement is placed on them.
            self.output[f"n_evt_single_mu_{prefix_cs}"] += int(ak.sum(n_mu_src == 1))
            self.output["time_at_ip"].fill(
                cat=ds_label, sel="1mu",
                val=ak.flatten(src[n_mu_src == 1].timeAtIpInOut))

            # ── original >=2-muon events (BEFORE the inTimeMuon requirement) ──
            ge2 = (n_mu_src >= 2)
            cs_orig = src[ge2]
            ev_orig = events[ge2]
            self.output[f"n_evt_ge2_orig_{prefix_cs}"] += len(cs_orig)

            if len(cs_orig) > 0:
                # timing of ALL muons in original >=2mu events (pre-inTimeMuon) — keeps the
                # out-of-time tail visible; identical to the previous version's 2mu_* plots.
                self.output["time_at_ip"].fill(cat=ds_label, sel="2mu_all",
                                               val=ak.flatten(cs_orig.timeAtIpInOut))
                self.output["time_at_ip"].fill(cat=ds_label, sel="2mu_upper",
                                               val=ak.flatten(cs_orig[cs_orig.phi > 0].timeAtIpInOut))
                self.output["time_at_ip"].fill(cat=ds_label, sel="2mu_lower",
                                               val=ak.flatten(cs_orig[cs_orig.phi < 0].timeAtIpInOut))

            # ── require inTimeMuon==True on NON-CANDIDATE muons (index > 0);
            #    the candidate (index 0) is always kept with the method C cuts only. ──
            idx = ak.local_index(cs_orig, axis=1)
            keep_itm = (idx == 0) | (cs_orig.inTimeMuon == True)
            cs_itm = cs_orig[keep_itm]
            n_after_itm = ak.num(cs_itm)

            # >=2mu events whose non-candidates are ALL out-of-time collapse to
            # candidate-only: no partner remains, so the veto cannot fire -> they SURVIVE.
            drop_to_1 = (n_after_itm == 1)
            self.output[f"n_evt_itm_to_1mu_{prefix_cs}"] += int(ak.sum(drop_to_1))
            if ak.sum(drop_to_1) > 0:
                self.output["time_at_ip"].fill(
                    cat=ds_label, sel="1mu_from_itm",
                    val=ak.flatten(cs_itm[drop_to_1].timeAtIpInOut))

            # events still having >=2 IN-TIME muons enter the veto
            still_ge2 = (n_after_itm >= 2)
            cs = cs_itm[still_ge2]
            ev2 = ev_orig[still_ge2]   # run/lumi/event aligned with cs (dup-survivor dump)

            if len(cs) > 0:
                self.output[f"n_evt_enter_{prefix_cs}"] += len(cs)
                self.output["cosA_study_input_mult"].fill(cat=ds_label, val=ak.num(cs))

                # ── STAGE 1: duplicate removal via outertrack/segment cosA (cosmic MC + NoBPTX) ──
                pr = ak.combinations(cs, 2, fields=["a", "b"])
                a, b = pr.a, pr.b
                dup = ((abs(a.eta - b.eta) < 0.01) &
                       (abs(a.delta_phi(b)) < 0.001))

                with np.errstate(divide="ignore", invalid="ignore"):
                    oa = _pmvec(a.outertrack_pt, a.outertrack_eta, a.outertrack_phi, a.mass)
                    ob = _pmvec(b.outertrack_pt, b.outertrack_eta, b.outertrack_phi, b.mass)
                    a_has = a.outertrack_pt > 0
                    b_has = b.outertrack_pt > 0
                    both  = a_has & b_has
                    onlyA = a_has & (~b_has)
                    onlyB = (~a_has) & b_has
                    A_flip = (a.phi > 0) != (a.outertrack_phi > 0)
                    B_flip = (b.phi > 0) != (b.outertrack_phi > 0)
                    elig = both | (onlyA & A_flip) | (onlyB & B_flip)
                    cos_oo = _cosA_vec(oa, ob)
                    cos_oi = _cosA_vec(oa, b)
                    cos_io = _cosA_vec(a, ob)
                    cos_out = ak.where(both, cos_oo,
                               ak.where(onlyA & A_flip, cos_oi,
                                ak.where(onlyB & B_flip, cos_io, np.nan)))

                    seg_dot = a.sumSegX*b.sumSegX + a.sumSegY*b.sumSegY + a.sumSegZ*b.sumSegZ
                    na = np.sqrt(a.sumSegX**2 + a.sumSegY**2 + a.sumSegZ**2)
                    nb = np.sqrt(b.sumSegX**2 + b.sumSegY**2 + b.sumSegZ**2)
                    both_seg = (a.nSeg > 0) & (b.nSeg > 0)
                    cos_seg = ak.where(both_seg & (na > 0) & (nb > 0), seg_dot/(na*nb), np.nan)

                rem_out  = dup & elig & (cos_out < -0.8)
                rem_seg  = dup & (~rem_out) & (cos_seg < -0.8)
                dup_surv = dup & (~rem_out) & (~rem_seg)

                self.output[f"n_dup_pairs_total_{prefix_cs}"]     += int(ak.sum(ak.sum(dup, axis=1)))
                self.output[f"n_dup_removed_outer_{prefix_cs}"]   += int(ak.sum(ak.sum(rem_out, axis=1)))
                self.output[f"n_dup_removed_seg_{prefix_cs}"]     += int(ak.sum(ak.sum(rem_seg, axis=1)))
                self.output[f"n_dup_pairs_surviving_{prefix_cs}"] += int(ak.sum(ak.sum(dup_surv, axis=1)))

                # ── record duplicate pairs that SURVIVE the outertrack + segment cuts ──
                #    (still flagged as duplicates → dump run/lumi/event + kinematics +
                #     the outertrack/segment diagnostics that let them slip through)
                if ak.any(dup_surv):
                    run_p,  _ = ak.broadcast_arrays(ev2.run, a.pt)
                    lumi_p, _ = ak.broadcast_arrays(ev2.luminosityBlock, a.pt)
                    evt_p,  _ = ak.broadcast_arrays(ev2.event, a.pt)

                    def _flat_surv(x):
                        return ak.to_numpy(ak.flatten(x[dup_surv], axis=1))

                    _cols = [
                        _flat_surv(run_p), _flat_surv(lumi_p), _flat_surv(evt_p),
                        _flat_surv(a.pt),  _flat_surv(a.eta),  _flat_surv(a.phi),
                        _flat_surv(b.pt),  _flat_surv(b.eta),  _flat_surv(b.phi),
                        _flat_surv(cos_out), _flat_surv(cos_seg),
                        _flat_surv(a.outertrack_pt), _flat_surv(a.outertrack_eta), _flat_surv(a.outertrack_phi),
                        _flat_surv(b.outertrack_pt), _flat_surv(b.outertrack_eta), _flat_surv(b.outertrack_phi),
                        _flat_surv(a.nSeg), _flat_surv(b.nSeg), _flat_surv(elig),
                    ]
                    for (rr, ll, ee, apt, aeta, aphi, bpt, beta, bphi,
                         co, csg, aopt, aoeta, aophi, bopt, boeta, bophi, ans, bns, el) in zip(*_cols):
                        self.output[f"dup_surv_rows_{prefix_cs}"].append((
                            int(rr), int(ll), int(ee),
                            float(apt), float(aeta), float(aphi),
                            float(bpt), float(beta), float(bphi),
                            float(co), float(csg),
                            float(aopt), float(aoeta), float(aophi),
                            float(bopt), float(boeta), float(bophi),
                            int(ans), int(bns), bool(el),
                        ))

                ev_veto_dup = ak.any(rem_out | rem_seg, axis=1)

                cs_dup = cs[~ev_veto_dup]
                self.output[f"n_evt_after_dup_{prefix_cs}"] += len(cs_dup)

                # ── STAGE 2: standard cosA < -0.99 (candidate vs each retained muon) ──
                if len(cs_dup) > 0:
                    lead2 = cs_dup[:, 0]     # = the method C candidate
                    sub2  = cs_dup[:, 1:]
                    dot = lead2.px*sub2.px + lead2.py*sub2.py + lead2.pz*sub2.pz
                    den = lead2.p * sub2.p
                    cosA = ak.where(den != 0, dot/den, -1000.0)
                    self.output["cosA_study_lead_vs_sub"].fill(cat=ds_label, val=ak.flatten(cosA))

                    ev_bb = ak.any(cosA < -0.99, axis=1)
                    cs_cosA = cs_dup[~ev_bb]
                    n_after_cosA = len(cs_cosA)
                    self.output[f"n_evt_after_cosA_{prefix_cs}"] += n_after_cosA

                    # ── STAGE 3: dt cut (ndof > 7 on the two dt leaders ONLY) ──
                    n_dt_veto = 0
                    if n_after_cosA > 0:
                        self.output["cosA_study_surviving_mult"].fill(cat=ds_label, val=ak.num(cs_cosA))

                        up_all = cs_cosA[cs_cosA.phi > 0]
                        lo_all = cs_cosA[cs_cosA.phi < 0]
                        has_both = (ak.num(up_all) >= 1) & (ak.num(lo_all) >= 1)

                        ev_dt_veto = np.zeros(n_after_cosA, dtype=bool)
                        if ak.sum(has_both) > 0:
                            ub = up_all[has_both]
                            lb = lo_all[has_both]
                            up = ub[ak.argsort(ub.pt, axis=1, ascending=False)][:, 0]
                            lo = lb[ak.argsort(lb.pt, axis=1, ascending=False)][:, 0]

                            ndof_ok = (up.timeNDof > 7) & (lo.timeNDof > 7)
                            dt = up.timeAtIpInOut - lo.timeAtIpInOut
                            self.output["cosA_study_surviving_dt"].fill(cat=ds_label, val=dt)

                            dt_veto = ndof_ok & (dt < -20)
                            n_dt_veto = int(ak.sum(dt_veto))
                            self.output["cosA_dt_study_surviving_dt"].fill(cat=ds_label, val=dt[~dt_veto])
                            ev_dt_veto[ak.to_numpy(has_both)] = ak.to_numpy(dt_veto)

                        # time of muons in >=2-mu events surviving the FULL veto (dup + cosA + dt)
                        cs_final = cs_cosA[~ev_dt_veto]
                        self.output["time_at_ip"].fill(cat=ds_label, sel="2mu_after_veto",
                                                       val=ak.flatten(cs_final.timeAtIpInOut))

                        # only "both-hemisphere" events can be dt-vetoed; the rest survive
                        self.output[f"n_evt_after_dt_{prefix_cs}"] += (n_after_cosA - n_dt_veto)

        # ── Signal: cosA + dt (fills the surviving-dt histogram only) ──
        if is_signal:
            sig_mask = (n_dismuons >= 2)
            if ak.sum(sig_mask) > 0:
                # METHOD C: index 0 is the candidate and already passed
                # pt/eta/mediumId/iso/dxy — keep candidate-first order
                # (no pT re-sort, no quality re-check).
                sig_sorted = dis_muons[sig_mask]

                if len(sig_sorted) > 0:
                    sig_lead = sig_sorted[:, 0]     # = the method C candidate
                    sig_sub  = sig_sorted[:, 1:]
                    dot_sig = sig_lead.px*sig_sub.px + sig_lead.py*sig_sub.py + sig_lead.pz*sig_sub.pz
                    denom_sig = sig_lead.p * sig_sub.p
                    cosA_sig = ak.where(denom_sig != 0, dot_sig / denom_sig, -1000.0)

                    event_has_cosmic_sig = ak.any(cosA_sig < -0.99, axis=1)
                    surviving_sig = sig_sorted[~event_has_cosmic_sig]

                    if len(surviving_sig) > 0:
                        sig_upper_all = surviving_sig[surviving_sig.phi > 0]
                        sig_lower_all = surviving_sig[surviving_sig.phi < 0]
                        sig_has_both = (ak.num(sig_upper_all) >= 1) & (ak.num(sig_lower_all) >= 1)

                        if ak.sum(sig_has_both) > 0:
                            su = sig_upper_all[sig_has_both]
                            sl = sig_lower_all[sig_has_both]
                            sig_upper = su[ak.argsort(su.pt, axis=1, ascending=False)][:, 0]
                            sig_lower = sl[ak.argsort(sl.pt, axis=1, ascending=False)][:, 0]

                            sig_ndof_ok = (sig_upper.timeNDof > 7) & (sig_lower.timeNDof > 7)
                            sig_dt = sig_upper.timeAtIpInOut - sig_lower.timeAtIpInOut
                            self.output["cosA_study_surviving_dt"].fill(cat=ds_label, val=sig_dt[sig_ndof_ok])

        return self.output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    cosmic_pkl = "scripts/samples/Run3_Summer22_chs_AK4PFCands_v21_DTTrigCalib/Cosmic_DTTrigCalib_preprocessed.pkl"
    print(f"Loading preprocessed Cosmics from {cosmic_pkl}...")
    with open(cosmic_pkl, "rb") as f:
        combined_runnable = pickle.load(f)

    nobptx_pkl = "samples/Summer22_CHS_v21_Cosmic/NoBPTX_preprocessed.pkl"
    print(f"Loading preprocessed NoBPTX from {nobptx_pkl}...")
    with open(nobptx_pkl, "rb") as f:
        nobptx_runnable = pickle.load(f)
    combined_runnable.update(nobptx_runnable)

    signal_pkl = "samples/Signal_Samples/Stau_300_100mm_preprocessed.pkl"
    print(f"Loading preprocessed Signal from {signal_pkl}...")
    with open(signal_pkl, "rb") as f:
        signal_runnable = pickle.load(f)
    combined_runnable.update(signal_runnable)

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

    print("=" * 80)
    print(" COSMIC REMOVAL CUTFLOW")
    print("=" * 80)

    for ds_name, prefix in [("COSMIC MC", "cosmic"), ("NoBPTX DATA", "nobptx")]:
        dtot  = out[f"n_dup_pairs_total_{prefix}"]
        dout  = out[f"n_dup_removed_outer_{prefix}"]
        dseg  = out[f"n_dup_removed_seg_{prefix}"]
        dsurv = out[f"n_dup_pairs_surviving_{prefix}"]
        extra = "   (sanity check: ~145 expected)" if prefix == "cosmic" else ""
        print(f"\n  {ds_name} — removing duplicate muon pairs")
        print(f"  (each pair can be removed by only one cut: outertrack is tried first, then segment)")
        print(f"    muon pairs flagged as duplicates              : {dtot}")
        print(f"    [1a] removed: outertrack cos(angle) < -0.8    : {dout}")
        print(f"    [1b] removed: segment   cos(angle) < -0.8     : {dseg}")
        print(f"    duplicate pairs still remaining afterwards    : {dsurv}{extra}")

    print("\n" + "-" * 80)
    print("  CUTS APPLIED — and exactly which muon each one acts on")
    print("-" * 80)
    print("  METHOD C: EVERY muon is tested against cuts (a)-(d); the CANDIDATE is the")
    print("  highest-pT muon that passes ALL of them (it need NOT be the highest-pT muon")
    print("  of the event). The event is kept only if a candidate exists. All other muons")
    print("  are RETAINED with no cuts (their only requirement is inTimeMuon, further")
    print("  down), and the candidate is placed at index 0 of the muon list:")
    print("      (a) candidate muon  pt > 30 GeV")
    print("      (b) candidate muon  |eta| < 2.4")
    print("      (c) candidate muon  mediumId == True")
    print("      (d) candidate muon  pfRelIso03_all < 0.18")
    print("      (signal only)       candidate must also have 0.1 < |dxy| < 10 cm")
    print("      (NoBPTX only)     event has npvsGood == 0  (no good primary vertex)")
    print("  inTimeMuon requirement (applied BEFORE the three veto stages below):")
    print("      every NON-CANDIDATE muon must have inTimeMuon == True. The candidate is")
    print("      always kept and is NOT required to be in-time. Single-muon events have")
    print("      no other muon, so nothing about them changes.")
    print("      If a two-or-more-muon event has ALL of its non-candidate muons out-of-time,")
    print("      those are removed and only the candidate is left; that event can no")
    print("      longer form a muon pair, so it is KEPT (there is nothing to veto).")
    print("  Three cosmic-veto stages, applied only to events that still have >=2 in-time muons:")
    print("      [1] a duplicate pair (|deta|<0.01, |dphi|<0.001) is removed if")
    print("          its outertrack cos(angle) < -0.8, otherwise if its segment cos(angle) < -0.8")
    print("      [2] cos(angle) < -0.99 between the candidate and ANY other retained muon")
    print("      [3] dt < -20 ns between the highest-pT muon in the upper half (phi>0) and")
    print("          the highest-pT muon in the lower half (phi<0), applied only when BOTH")
    print("          of those two muons have timeNDof > 7")
    print("-" * 80)

    for ds_name, prefix in [("COSMIC MC", "cosmic"), ("NoBPTX DATA", "nobptx")]:
        n1     = out[f"n_evt_single_mu_{prefix}"]     # events with exactly one muon
        g2o    = out[f"n_evt_ge2_orig_{prefix}"]      # events with >=2 muons (before inTimeMuon)
        itm1   = out[f"n_evt_itm_to_1mu_{prefix}"]    # >=2mu events left with only the candidate
        e0     = out[f"n_evt_enter_{prefix}"]         # >=2 in-time-muon events entering the veto
        e1     = out[f"n_evt_after_dup_{prefix}"]
        e2     = out[f"n_evt_after_cosA_{prefix}"]
        e3     = out[f"n_evt_after_dt_{prefix}"]
        n_proc = out[f"n_events_total_{prefix}"]
        surv_total  = n1 + itm1 + e3
        enter_total = n1 + g2o                        # every event with a method C candidate (has >=1 muon)
        print(f"\n  {ds_name} — how many cosmic events remain after each cut:")
        if prefix == "nobptx":
            print(f"    (information only, NOT removed here) NoBPTX events whose candidate muon")
            print(f"    has a near-identical duplicate track      : {out['n_events_with_dup_nobptx']}")
        print(f"    events with exactly ONE muon (skip every veto stage, always kept) : {n1}")
        print(f"    events with TWO OR MORE muons (before the inTimeMuon requirement) : {g2o}")
        print(f"    -- now require inTimeMuon == True on every non-candidate muon --")
        print(f"    ... of those, events now left with ONLY the candidate")
        print(f"        (all their non-candidate muons were out-of-time) -> KEPT     : {itm1}")
        print(f"    ... events that STILL have >=2 in-time muons -> enter the veto    : {e0}")
        print(f"        (consistency check: {itm1} + {e0} = {itm1 + e0}, should equal {g2o})")
        print(f"    [1] remaining after duplicate removal (outertrack + segment cos(angle)) : {e1}   (removed {e0 - e1})")
        print(f"    [2] remaining after cos(angle) < -0.99 (candidate vs retained)         : {e2}   (removed {e1 - e2})")
        print(f"    [3] remaining after dt < -20 ns (both muons timeNDof > 7)              : {e3}   (removed {e2 - e3})")
        print(f"    ---------------- TOTAL cosmic events remaining ----------------")
        print(f"    {n1} (one muon) + {itm1} (only candidate left) + {e3} (>=2 muons, passed veto) = {surv_total}")
        if enter_total > 0:
            print(f"    as a fraction of events with a method C candidate    : {surv_total}/{enter_total} = {100.0*surv_total/enter_total:.2f}%")
        if n_proc > 0:
            print(f"    as a fraction of ALL events processed                 : {surv_total}/{n_proc} = {100.0*surv_total/n_proc:.2f}%")
    print("=" * 80 + "\n")

    print("=" * 80)
    print(" TOTAL EVENTS PROCESSED (one-muon fraction)")
    print("=" * 80)
    for ds_name, prefix in [("COSMIC MC", "cosmic"), ("NoBPTX DATA", "nobptx")]:
        n_total       = out[f"n_events_total_{prefix}"]
        n_single_raw  = out[f"n_evt_single_mu_raw_{prefix}"]
        n_single_cut  = out[f"n_evt_single_mu_{prefix}"]
        print(f"\n  {ds_name}:")
        print(f"    total events processed                                    : {n_total}")
        print(f"    events with exactly one muon, before any selection        : {n_single_raw}")
        print(f"    events with exactly one muon, after method C selection    : {n_single_cut}")
        if n_total > 0:
            print(f"    one-muon fraction before selection : {n_single_raw}/{n_total} = {100.0*n_single_raw/n_total:.2f}%")
            print(f"    one-muon fraction after selection  : {n_single_cut}/{n_total} = {100.0*n_single_cut/n_total:.2f}%")
    print("=" * 80 + "\n")

    # ════════════════════════════════════════════════════════════════════════
    #  DUMP: duplicate pairs that survive the outertrack + segment cuts.
    #  Two SEPARATE files — one for cosmic MC, one for NoBPTX data — with
    #  run/lumi/event + kinematics so you can see why they still register as
    #  duplicates.
    # ════════════════════════════════════════════════════════════════════════
    _hdr = (
        "{:>8} {:>7} {:>13}  "
        "{:>8} {:>7} {:>7}  {:>8} {:>7} {:>7}  "
        "{:>8} {:>8}  {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}  {:>6} {:>6}  {:>5}\n"
    ).format("run", "lumi", "event",
             "a_pt", "a_eta", "a_phi", "b_pt", "b_eta", "b_phi",
             "cos_out", "cos_seg",
             "a_otpt", "a_oteta", "a_otphi", "b_otpt", "b_oteta", "b_otphi",
             "a_nSeg", "b_nSeg", "elig")

    def _write_dup_survivor_file(path, ds_name, rows):
        with open(path, "w") as fh:
            fh.write(f"{ds_name}: duplicate PAIRS that survive the outertrack + segment cos(angle) < -0.8 cuts.\n")
            fh.write("These still look like duplicate / back-to-back muons after the outertrack + segment\n")
            fh.write("removal, so they keep registering as duplicates. One row per surviving pair (muons a and b).\n")
            fh.write("NOTE: muon selection uses METHOD C (candidate = highest-pT muon passing all quality\n")
            fh.write("cuts; all other muons retained). With the inTimeMuon requirement applied to\n")
            fh.write("non-candidates, only in-time muons reach this stage.\n")
            fh.write("Diagnostics that explain why STAGE 1 missed them:\n")
            fh.write("  cos_out = outertrack cos(angle)  (nan = pair not outertrack-eligible / no outer track)\n")
            fh.write("  cos_seg = segment    cos(angle)  (nan = a or b has nSeg == 0)\n")
            fh.write("  a_otpt/b_otpt = outertrack_pt    (<= 0 means no standalone outer track)\n")
            fh.write("  a_nSeg/b_nSeg = # muon segments,  elig = outertrack-eligible flag\n")
            fh.write("A pair is only removed when elig and cos_out < -0.8, or when cos_seg < -0.8.\n")
            fh.write("\n" + "=" * 118 + "\n")
            fh.write(f" {ds_name} — {len(rows)} surviving duplicate pair(s)\n")
            fh.write("=" * 118 + "\n")
            fh.write(_hdr)
            for (rr, ll, ee, apt, aeta, aphi, bpt, beta, bphi,
                 co, csg, aopt, aoeta, aophi, bopt, boeta, bophi, ans, bns, el) in rows:
                fh.write(
                    ("{:>8d} {:>7d} {:>13d}  "
                     "{:>8.3f} {:>7.3f} {:>7.3f}  {:>8.3f} {:>7.3f} {:>7.3f}  "
                     "{:>8.3f} {:>8.3f}  {:>8.3f} {:>8.3f} {:>8.3f}  {:>8.3f} {:>8.3f} {:>8.3f}  {:>6d} {:>6d}  {:>5}\n").format(
                        rr, ll, ee, apt, aeta, aphi, bpt, beta, bphi,
                        co, csg, aopt, aoeta, aophi, bopt, boeta, bophi, ans, bns, str(el))
                )

        print(f"Wrote {len(rows)} {ds_name} surviving duplicate pair(s) to: {os.path.abspath(path)}")

    _write_dup_survivor_file("dup_survivor_events_MC.txt",   "COSMIC MC",   out["dup_surv_rows_cosmic"])
    _write_dup_survivor_file("dup_survivor_events_data.txt", "NoBPTX DATA", out["dup_surv_rows_nobptx"])
    print()

    '''
    # ── cosA study plots ──
    COSA_OUTPUT_DIR = "cosA_study_plots"
    os.makedirs(COSA_OUTPUT_DIR, exist_ok=True)
    cosA_overlay_plots = ["cosA_study_surviving_dt", "cosA_dt_study_surviving_dt"]
    for key in cosA_overlay_plots:
        save_comparison_overlay(out[key], key, "cosA_study_", COSA_OUTPUT_DIR,
                                title_suffix="(Cosmics MC vs NoBPTX Data)",
                                filename_suffix="MC_vs_Data", normalize=True)

    # ── Standalone / Global / Tracker muon counts ──
    OUTPUT_DIR = "single_muon_signal_vs_cosmic_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_muon_type_counts(out["muon_type"], "muon_type", "single_muon_", OUTPUT_DIR,
                          title_suffix="(300 GeV 100 mm vs Cosmics)",
                          filename_suffix="Stau_300_100mm_overlay")
    '''

    # ── timeAtIpInOut / inTimeMuon study plots ──
    TIME_OUTPUT_DIR = "time_study_plots"
    os.makedirs(TIME_OUTPUT_DIR, exist_ok=True)
    for sel in ["1mu", "2mu_all", "2mu_upper", "2mu_lower", "2mu_after_veto", "1mu_from_itm"]:
        save_time_overlay(out["time_at_ip"], sel, "ITM_study_", TIME_OUTPUT_DIR,
                          title_suffix="(Cosmic MC vs NoBPTX Data)", normalize=True)

    print("Done!")