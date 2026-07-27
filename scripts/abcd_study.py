#!/usr/bin/env python3
"""
abcd_study.py — standalone ABCD study for the cosmic background estimate.

WHAT THIS DOES
==============
Implements the ABCD method (Buttinger, "Background Estimation with the ABCD
Method") for the cosmic background, in two independent categories:

  CATEGORY 1 — events with >= 2 DisMuons (after duplicate removal)
      x-axis : cosA_min = most back-to-back cos(opening angle) between the
               method-C candidate and any retained partner muon
      y-axis : dt = timeAtIpInOut(upper leader) - timeAtIpInOut(lower leader),
               defined only when both hemisphere leaders have timeNDof > 7
      Regions (cuts COSA_CUT = -0.99, DT_CUT = -20 ns):
          A (signal-like) : cosA_min > -0.99  AND  dt > -20   <- SR leakage
          B               : cosA_min > -0.99  AND  dt < -20
          C               : cosA_min < -0.99  AND  dt > -20
          D               : cosA_min < -0.99  AND  dt < -20
      Prediction:  N_A = N_B * N_C / N_D

  CATEGORY 2 — events with exactly 1 DisMuon (the dominant leftover!)
      x-axis : timeAtIpInOut of the muon (needs timeNDof > 7)
      y-axis : |dz| of the muon
      Regions (cuts T1MU_CUT = -15 ns, DZ_CUT = 10 cm):
          A (signal-like) : t > -15  AND  |dz| < 10
          B               : t < -15  AND  |dz| < 10
          C               : t > -15  AND  |dz| > 10
          D               : t < -15  AND  |dz| > 10
      Filled SEPARATELY for upper (phi>0) and lower (phi<0) hemispheres,
      because timing only discriminates for upper (inward-going) legs.
      If the lower-hemisphere plane shows no timing separation, that
      category needs a different axis (opposite-hemisphere activity).

  Events that cannot be placed on a plane are counted honestly:
      2mu same-hemisphere / ndof-fail  -> "no-dt" bucket (cosA_min histogrammed)
      1mu with timeNDof == 0..7        -> "no-time" bucket
  These are populations the ABCD cannot reach and need their own strategy.

  EXTRAS
  - Transfer-factor stability scans (guide sec 2.4): closure ratio vs the dt
    cut position and vs the cosA cut position.
  - In-time-fraction vs cosA profile: a flat profile = the independence
    assumption (guide eq. 1) holds.
  - |dphi| vs cosA 2D per dataset: beam-halo shows up at (dphi~0, cosA~-0.6
    to -0.95); cosmics at (dphi~pi, cosA~-1).
  - PFMET for 1mu vs >=2mu events (does MET anti-select 2-leg cosmics?).
  - Cosmic-shower diagnostics: collinear (cosA > SHOWER_COSA, non-duplicate)
    pairs SURVIVING the outertrack+segment tests are dumped with muon
    multiplicity, timing spread, and a run:lumi:event pick-list for
    Fireworks / edmPickEvents.py event displays.

DATASET ROLES (from dataset name in the preprocessed pkl):
  "Cosmic*"/"LooseMu*" -> CosmicMC   : closure test + diagnostics (A unblinded)
  "*NoBPTX*"           -> NoBPTX     : THE closure sample (pure cosmic, A unblinded)
  "*Stau*"/"*Signal*"  -> Signal     : contamination check ONLY (guide checklist #3:
                                       how much signal lands in B/C/D?)
  anything else        -> Collision  : where the real prediction is made.
                                       Region A is BLINDED (BLIND_COLLISION_A).

RUN:  python3 abcd_study.py     (edit the pkl paths at the bottom first)
Outputs land in abcd_output/ : printed report, pdf plots, dumps, raw pickle.
"""
import os
import pickle
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import coffea.nanoevents.methods.vector as vector

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG — every cut the ABCD depends on lives here
# ════════════════════════════════════════════════════════════════════════════
COSA_CUT = -0.99     # back-to-back boundary  (must be a multiple of 0.005)
DT_CUT   = -20.0     # out-of-time boundary for the dimuon dt   [ns] (integer)
T_NDOF   = 7         # timing valid when timeNDof > T_NDOF  (same as the veto)
T1MU_CUT = -15.0     # out-of-time boundary for the single-muon time [ns] (integer)
DZ_CUT   = 10.0      # large-|dz| boundary for the single-muon plane [cm] (integer)
SHOWER_COSA = 0.90   # collinear-pair threshold feeding the shower study
BLIND_COLLISION_A = True   # NEVER flip this casually — group sign-off first
OUT_DIR = "abcd_output"

# ════════════════════════════════════════════════════════════════════════════
#  helpers (same idioms as cosmics_final.py)
# ════════════════════════════════════════════════════════════════════════════
def get_Lxy(genvistau):
    vx = genvistau.parent.vx - genvistau.parent.distinctParent.vx
    vy = genvistau.parent.vy - genvistau.parent.distinctParent.vy
    return np.sqrt(vx ** 2 + vy ** 2)

def _cosA_vec(u, v):
    denom = u.p * v.p
    return ak.where(denom != 0, (u.px * v.px + u.py * v.py + u.pz * v.pz) / denom, np.nan)

def _np(x):
    return ak.to_numpy(x)

def abcd_predict(B, C, D):
    """N_A prediction with its 'syst' from B/C/D statistics (guide eq. 3-4).
    Returns (pred, syst). stat on the prediction is sqrt(pred) — the Poisson
    scatter expected on the observed A count itself."""
    if D <= 0 or B <= 0 or C <= 0:
        return np.nan, np.nan
    pred = B * C / D
    syst = pred * np.sqrt(1.0 / B + 1.0 / C + 1.0 / D)
    return pred, syst

# ════════════════════════════════════════════════════════════════════════════
#  processor
# ════════════════════════════════════════════════════════════════════════════
PREFIXES = ("cosmic", "nobptx", "signal", "data")

class ABCDProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            # fine-binned planes; ALL region arithmetic happens offline in the
            # analysis section, so cuts can be re-scanned without re-running.
            # cosA axis: width 0.005 -> cuts at multiples of 0.005 are bin edges.
            # dt / t axes: width 1 ns -> integer cuts are bin edges.
            "plane2mu": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(402, -1.005, 1.005, name="cosa", label=r"$\cos\alpha_{min}$ (candidate vs most back-to-back partner)"),
                axis.Regular(120, -60, 60, name="dt", label=r"$\Delta t$ (upper - lower) [ns]"),
            ),
            "plane1mu": Hist(
                axis.StrCategory([], name="cat", growth=True),
                axis.StrCategory([], name="hemi", growth=True),
                axis.Regular(200, -100, 100, name="t", label="timeAtIpInOut [ns]"),
                axis.Regular(100, 0, 100, name="dz", label="|dz| [cm]"),
            ),
            # diagnostics
            "cosa_nodt": Hist(              # 2mu events the dt axis can't classify
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(402, -1.005, 1.005, name="cosa", label=r"$\cos\alpha_{min}$ (no valid $\Delta t$)"),
            ),
            "dphi_vs_cosa": Hist(           # beam-halo check
                axis.StrCategory([], name="cat", growth=True),
                axis.Regular(100, -1.005, 1.005, name="cosa", label=r"$\cos\alpha_{min}$"),
                axis.Regular(64, 0, np.pi, name="dphi", label=r"$|\Delta\phi|$ (candidate, partner)"),
            ),
            "tio_vs_toi_1mu": Hist(         # single-leg in-out vs out-in (if branch exists)
                axis.StrCategory([], name="cat", growth=True),
                axis.StrCategory([], name="hemi", growth=True),
                axis.Regular(80, -80, 80, name="tio", label="timeAtIpInOut [ns]"),
                axis.Regular(80, -80, 80, name="toi", label="timeAtIpOutIn [ns]"),
            ),
            "met": Hist(                    # MET anti-selection check
                axis.StrCategory([], name="cat", growth=True),
                axis.StrCategory([], name="kind", growth=True),   # 1mu / ge2mu
                axis.Regular(100, 0, 500, name="val", label="PFMET [GeV]"),
            ),
            "mult": Hist(                   # shower study: muon multiplicity
                axis.StrCategory([], name="cat", growth=True),
                axis.StrCategory([], name="kind", growth=True),   # all / shower
                axis.Regular(15, 0, 15, name="val", label="DisMuons per event"),
            ),
        }
        for p in PREFIXES:
            self.output[f"n_total_{p}"] = 0          # events processed
            self.output[f"n_cand_{p}"] = 0           # events with a method-C candidate
            self.output[f"n_1mu_{p}"] = 0
            self.output[f"n_1mu_notime_up_{p}"] = 0  # 1mu, timeNDof<=T_NDOF, phi>0
            self.output[f"n_1mu_notime_lo_{p}"] = 0
            self.output[f"n_ge2_{p}"] = 0
            self.output[f"n_2mu_same_hemi_{p}"] = 0  # no (phi>0, phi<0) pair -> no dt
            self.output[f"n_2mu_ndof_fail_{p}"] = 0  # dt exists but a leader fails ndof
            self.output[f"n_2mu_classified_{p}"] = 0
            self.output[f"n_shower_{p}"] = 0         # events with a surviving collinear pair
            self.output[f"shower_rows_{p}"] = []

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")
        is_cosmic = "Cosmic" in dataset or dataset.startswith("LooseMu") or dataset == "test_cosmics_calib"
        is_nobptx = "NoBPTX" in dataset
        is_signal = (not is_cosmic and not is_nobptx) and ("Stau" in dataset or "Signal" in dataset)
        is_data = not (is_cosmic or is_nobptx or is_signal)
        p = "cosmic" if is_cosmic else "nobptx" if is_nobptx else "signal" if is_signal else "data"
        cat = {"cosmic": "CosmicMC", "nobptx": "NoBPTX", "signal": "Signal", "data": "Collision"}[p]
        has_gen = "GenPart" in events.fields

        self.output[f"n_total_{p}"] += len(events)

        # ── zip the DisMuon fields we need; optional branches only if present ──
        fields = {
            "pt": events.DisMuon.pt, "eta": events.DisMuon.eta,
            "phi": events.DisMuon.phi, "mass": events.DisMuon.mass,
            "charge": events.DisMuon.charge, "dxy": events.DisMuon.dxy,
            "dz": events.DisMuon.dz, "timeNDof": events.DisMuon.timeNDof,
            "timeAtIpInOut": events.DisMuon.timeAtIpInOut,
            "mediumId": events.DisMuon.mediumId,
            "pfRelIso03_all": events.DisMuon.pfRelIso03_all,
        }
        for f in ("timeAtIpOutIn", "outertrack_pt", "outertrack_eta",
                  "outertrack_phi", "sumSegX", "sumSegY", "sumSegZ", "nSeg"):
            if f in events.DisMuon.fields:
                fields[f] = events.DisMuon[f]
        events["DisMuon"] = ak.zip(fields, with_name="PtEtaPhiMLorentzVector",
                                   behavior=vector.behavior)

        # ── gen-level signal region (signal only; same as cosmics_final.py) ──
        if is_signal and has_gen:
            gpart = events.GenPart
            genvistau_Lxy = get_Lxy(events.GenVisTau)
            events["GenVisStauTaus"] = events.GenVisTau[
                (abs(events.GenVisTau.parent.pdgId) == 15) &
                (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
                events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy") &
                events.GenVisTau.parent.hasFlags("fromHardProcess") &
                (genvistau_Lxy < 100.0) &
                (events.GenVisTau.pt > 20) &
                (abs(events.GenVisTau.eta) < 2.4)
            ]
            events["GenMuon"] = gpart[(abs(gpart.pdgId) == 13) & gpart.hasFlags("isLastCopy")]
            events["GenMuon"] = events.GenMuon[
                (events.GenMuon.pt > 20) & (abs(events.GenMuon.eta) < 2.4) &
                (abs(events.GenMuon.distinctParent.distinctParent.pdgId) == 1000015)
            ]
            events["GenElectron"] = gpart[(abs(gpart.pdgId) == 11) & gpart.hasFlags("isLastCopy")]
            events["GenElectron"] = events.GenElectron[
                (events.GenElectron.pt > 20) & (abs(events.GenElectron.eta) < 2.4) &
                (abs(events.GenElectron.distinctParent.distinctParent.pdgId) == 1000015)
            ]
            events = events[
                (ak.num(events.GenVisStauTaus) == 1) &
                (ak.num(events.GenMuon) == 1) &
                (ak.num(events.GenElectron) == 0)
            ]

        # ── METHOD C candidate selection (identical to cosmics_final.py):
        #    candidate = highest-pT muon passing all quality cuts, moved to
        #    index 0; all other muons retained uncut. Signal AND collision data
        #    get the SR dxy window; cosmic/NoBPTX do not (max cosmic stats —
        #    checking the transfer factor WITH the dxy window is a follow-up). ──
        sorted_all = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
        pass_kin = (
            (sorted_all.pt > 30) & (abs(sorted_all.eta) < 2.4) &
            (sorted_all.mediumId == True) & (sorted_all.pfRelIso03_all < 0.18)
        )
        if is_signal or is_data:
            pass_kin = pass_kin & (abs(sorted_all.dxy) > 0.1) & (abs(sorted_all.dxy) < 10)
        li = ak.local_index(sorted_all.pt, axis=1)
        cand_idx = ak.firsts(li[pass_kin])
        has_cand = ~ak.is_none(cand_idx)
        order = ak.concatenate(
            [ak.singletons(cand_idx), li[li != ak.fill_none(cand_idx, -1)]], axis=1)
        reordered = sorted_all[order]
        keep, _ = ak.broadcast_arrays(has_cand, reordered.pt)
        events["DisMuon"] = reordered[keep]

        # ── NoBPTX: no good primary vertex (as in cosmics_final.py) ──
        if is_nobptx:
            events = events[events.PV.npvsGood == 0]

        # ── signal / collision data: reco SR jet + MET selection ──
        if (is_signal and has_gen) or is_data:
            charged_sel = events.Jet.constituents.pf.charge != 0
            jdxy = ak.where(
                ak.all(events.Jet.constituents.pf.charge == 0, axis=-1), -999,
                ak.flatten(events.Jet.constituents.pf[
                    ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)
                ].d0, axis=-1))
            jdxy = ak.fill_none(jdxy, -999)
            events["Jet"] = ak.with_field(events.Jet, jdxy, where="dxy")
            jets = events.Jet[
                (abs(events.Jet.eta) < 2.4) & (events.Jet.pt > 32) &
                (events.Jet.neHEF < 0.99) & (events.Jet.neEmEF < 0.9) &
                ((events.Jet.chMultiplicity + events.Jet.neMultiplicity) > 1) &
                (events.Jet.chMultiplicity > 0) & (events.Jet.muEF < 0.1) &
                (events.Jet.chEmEF < 0.8) &
                (events.Jet.disTauTag_score1 > 0.9) & (abs(events.Jet.dxy) > 0.02)
            ]
            events = events[(ak.num(jets) == 1) & (events.PFMET.pt > 105)]

        # ── require a candidate ──
        mu = events.DisMuon
        keep_ev = ak.num(mu) >= 1
        events = events[keep_ev]
        mu = events.DisMuon
        self.output[f"n_cand_{p}"] += len(events)
        if len(events) == 0:
            return self.output

        # ── duplicate removal vs the candidate (deta/dphi/charge criterion of
        #    cosmics_final.py). Unlike the veto study we DROP the ghosts for
        #    every dataset: region classification should see physical muons.
        #    A 2-muon event whose partner is a ghost becomes a 1-muon event. ──
        cand0 = mu[:, 0]
        is_dup = ((mu.charge * cand0.charge) > 0) & \
                 (abs(mu.eta - cand0.eta) < 0.01) & \
                 (abs(mu.delta_phi(cand0)) < 0.001) & \
                 (ak.local_index(mu, axis=1) > 0)
        mu = mu[~is_dup]
        n = ak.num(mu)

        # ── MET: does MET>105 anti-select 2-leg cosmics? (field-guarded) ──
        if "PFMET" in events.fields:
            self.output["met"].fill(cat=cat, kind="1mu",
                val=np.clip(_np(events.PFMET.pt[n == 1]), 0, 499.9))
            self.output["met"].fill(cat=cat, kind="ge2mu",
                val=np.clip(_np(events.PFMET.pt[n >= 2]), 0, 499.9))

        # ════════════════════════════════════════════════════════════════════
        #  CATEGORY 2 — exactly one DisMuon  (plane: timeAtIpInOut x |dz|)
        # ════════════════════════════════════════════════════════════════════
        m1 = ak.firsts(mu[n == 1])
        n1 = int(ak.sum(n == 1))
        self.output[f"n_1mu_{p}"] += n1
        if n1 > 0:
            upper = _np(m1.phi > 0)
            tvalid = _np(m1.timeNDof > T_NDOF)
            self.output[f"n_1mu_notime_up_{p}"] += int(np.sum(upper & ~tvalid))
            self.output[f"n_1mu_notime_lo_{p}"] += int(np.sum(~upper & ~tvalid))
            t_all = _np(m1.timeAtIpInOut)
            dz_all = np.abs(_np(m1.dz))
            for hemi, hmask in (("upper", upper), ("lower", ~upper)):
                sel = hmask & tvalid
                if np.sum(sel) > 0:
                    self.output["plane1mu"].fill(
                        cat=cat, hemi=hemi,
                        t=np.clip(t_all[sel], -99.5, 99.5),
                        dz=np.clip(dz_all[sel], 0.001, 99.5))
                    if "timeAtIpOutIn" in mu.fields:
                        toi = _np(m1.timeAtIpOutIn)
                        self.output["tio_vs_toi_1mu"].fill(
                            cat=cat, hemi=hemi,
                            tio=np.clip(t_all[sel], -79.9, 79.9),
                            toi=np.clip(toi[sel], -79.9, 79.9))

        # ════════════════════════════════════════════════════════════════════
        #  CATEGORY 1 — >= 2 DisMuons  (plane: cosA_min x dt)
        # ════════════════════════════════════════════════════════════════════
        ge2 = n >= 2
        cs = mu[ge2]
        ev2 = events[ge2]
        N = len(cs)
        self.output[f"n_ge2_{p}"] += N
        if N == 0:
            return self.output

        cand = cs[:, 0]
        others = cs[:, 1:]
        cosa_all = _cosA_vec(cand, others)
        cosa_min = ak.min(cosa_all, axis=1)          # most back-to-back partner
        idx_min = ak.argmin(cosa_all, axis=1, keepdims=True)
        partner = ak.firsts(others[idx_min])
        dphi_bb = abs(cand.delta_phi(partner))
        self.output["dphi_vs_cosa"].fill(
            cat=cat, cosa=np.clip(_np(cosa_min), -1.004, 1.004), dphi=_np(dphi_bb))
        self.output["mult"].fill(cat=cat, kind="all", val=np.clip(_np(ak.num(cs)), 0, 14))

        # dt from the hemisphere leaders (same algorithm as the veto)
        up_all = cs[cs.phi > 0]
        lo_all = cs[cs.phi < 0]
        has_both = (ak.num(up_all) >= 1) & (ak.num(lo_all) >= 1)
        dt_full = np.full(N, np.nan)
        ndof_ok_full = np.zeros(N, dtype=bool)
        if ak.sum(has_both) > 0:
            ub = up_all[has_both]
            lb = lo_all[has_both]
            up = ub[ak.argsort(ub.pt, axis=1, ascending=False)][:, 0]
            lo = lb[ak.argsort(lb.pt, axis=1, ascending=False)][:, 0]
            hb = _np(has_both)
            dt_full[hb] = _np(up.timeAtIpInOut - lo.timeAtIpInOut)
            ndof_ok_full[hb] = _np((up.timeNDof > T_NDOF) & (lo.timeNDof > T_NDOF))

        hb_np = _np(has_both)
        defined = hb_np & ndof_ok_full
        self.output[f"n_2mu_same_hemi_{p}"] += int(np.sum(~hb_np))
        self.output[f"n_2mu_ndof_fail_{p}"] += int(np.sum(hb_np & ~ndof_ok_full))
        self.output[f"n_2mu_classified_{p}"] += int(np.sum(defined))

        cosa_np = _np(cosa_min)
        if np.sum(defined) > 0:
            self.output["plane2mu"].fill(
                cat=cat,
                cosa=np.clip(cosa_np[defined], -1.004, 1.004),
                dt=np.clip(dt_full[defined], -59.5, 59.5))
        if np.sum(~defined) > 0:
            self.output["cosa_nodt"].fill(
                cat=cat, cosa=np.clip(cosa_np[~defined], -1.004, 1.004))

        # ── SHOWER STUDY: collinear (non-dup) pairs surviving outer+segment ──
        pr = ak.combinations(cs, 2, fields=["a", "b"])
        a, b = pr.a, pr.b
        cos_ab = _cosA_vec(a, b)
        dup_pair = ((a.charge * b.charge) > 0) & \
                   (abs(a.eta - b.eta) < 0.01) & (abs(a.delta_phi(b)) < 0.001)
        coll = (cos_ab > SHOWER_COSA) & (~dup_pair)

        has_outer = "outertrack_pt" in cs.fields
        has_seg = "nSeg" in cs.fields
        if has_outer:
            with np.errstate(divide="ignore", invalid="ignore"):
                oa = ak.zip({"pt": a.outertrack_pt, "eta": a.outertrack_eta,
                             "phi": a.outertrack_phi, "mass": a.mass},
                            with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
                ob = ak.zip({"pt": b.outertrack_pt, "eta": b.outertrack_eta,
                             "phi": b.outertrack_phi, "mass": b.mass},
                            with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
                a_has, b_has = a.outertrack_pt > 0, b.outertrack_pt > 0
                both = a_has & b_has
                onlyA = a_has & (~b_has)
                onlyB = (~a_has) & b_has
                A_flip = (a.phi > 0) != (a.outertrack_phi > 0)
                B_flip = (b.phi > 0) != (b.outertrack_phi > 0)
                elig = both | (onlyA & A_flip) | (onlyB & B_flip)
                cos_out = ak.where(both, _cosA_vec(oa, ob),
                           ak.where(onlyA & A_flip, _cosA_vec(oa, b),
                            ak.where(onlyB & B_flip, _cosA_vec(a, ob), np.nan)))
        else:
            elig = cos_ab > 2          # all-False jagged bool
            cos_out = cos_ab * np.nan
        if has_seg:
            with np.errstate(divide="ignore", invalid="ignore"):
                seg_dot = a.sumSegX * b.sumSegX + a.sumSegY * b.sumSegY + a.sumSegZ * b.sumSegZ
                na = np.sqrt(a.sumSegX ** 2 + a.sumSegY ** 2 + a.sumSegZ ** 2)
                nb = np.sqrt(b.sumSegX ** 2 + b.sumSegY ** 2 + b.sumSegZ ** 2)
                cos_seg = ak.where((a.nSeg > 0) & (b.nSeg > 0) & (na > 0) & (nb > 0),
                                   seg_dot / (na * nb), np.nan)
            nseg_pair = a.nSeg + b.nSeg
        else:
            cos_seg = cos_ab * np.nan
            nseg_pair = ak.values_astype(cos_ab * 0 - 1, np.int64)

        rem_out = coll & elig & (cos_out < -0.8)
        rem_seg = coll & (~rem_out) & (cos_seg < -0.8)
        surv = coll & (~rem_out) & (~rem_seg)      # the unexplained collinear pairs

        shower_ev = ak.any(surv, axis=1)
        n_sh = int(ak.sum(shower_ev))
        self.output[f"n_shower_{p}"] += n_sh
        if n_sh > 0:
            self.output["mult"].fill(cat=cat, kind="shower",
                                     val=np.clip(_np(ak.num(cs[shower_ev])), 0, 14))
            # timing coherence: spread of timeAtIpInOut among timed muons
            tt = cs.timeAtIpInOut[cs.timeNDof > T_NDOF]
            tspread = ak.fill_none(ak.max(tt, axis=1) - ak.min(tt, axis=1), np.nan)
            nmu_ev = ak.num(cs)

            run_p, _ = ak.broadcast_arrays(ev2.run, a.pt)
            lumi_p, _ = ak.broadcast_arrays(ev2.luminosityBlock, a.pt)
            evt_p, _ = ak.broadcast_arrays(ev2.event, a.pt)
            nmu_p, _ = ak.broadcast_arrays(nmu_ev, a.pt)
            tsp_p, _ = ak.broadcast_arrays(tspread, a.pt)
            dt_p, _ = ak.broadcast_arrays(ak.Array(dt_full), a.pt)

            def _f(x):
                return ak.to_numpy(ak.flatten(x[surv], axis=1))

            for (rr, ll, ee, nm, cab, co, csg, ns, ts, dtv) in zip(
                    _f(run_p), _f(lumi_p), _f(evt_p), _f(nmu_p), _f(cos_ab),
                    _f(cos_out), _f(cos_seg), _f(nseg_pair), _f(tsp_p), _f(dt_p)):
                self.output[f"shower_rows_{p}"].append((
                    int(rr), int(ll), int(ee), int(nm), float(cab),
                    float(co), float(csg), int(ns), float(ts), float(dtv)))

        return self.output

    def postprocess(self, accumulator):
        return accumulator


# ════════════════════════════════════════════════════════════════════════════
#  offline analysis: region counting, prediction, closure, scans, plots
# ════════════════════════════════════════════════════════════════════════════
def regions_from_2d(h, sel, xname, yname, xcut, ycut, x_sig_is_high, y_sig_is_high):
    """A = both signal-like, B = x sig-like & y cosmic-like,
       C = x cosmic-like & y sig-like, D = both cosmic-like.  pred = B*C/D."""
    hh = h[sel]
    vals = hh.values()
    xc = hh.axes[xname].centers
    yc = hh.axes[yname].centers
    xs = (xc > xcut) if x_sig_is_high else (xc < xcut)
    ys = (yc > ycut) if y_sig_is_high else (yc < ycut)
    A = vals[np.ix_(xs, ys)].sum()
    B = vals[np.ix_(xs, ~ys)].sum()
    C = vals[np.ix_(~xs, ys)].sum()
    D = vals[np.ix_(~xs, ~ys)].sum()
    return A, B, C, D


def print_abcd(tag, A, B, C, D, blind=False):
    pred, syst = abcd_predict(B, C, D)
    print(f"\n  {tag}")
    print(f"    B (sig-like x, cosmic-like y) : {B:>10.0f}")
    print(f"    C (cosmic-like x, sig-like y) : {C:>10.0f}")
    print(f"    D (both cosmic-like)          : {D:>10.0f}")
    if np.isnan(pred):
        print("    prediction    : UNDEFINED (a control region is empty)")
        return
    stat = np.sqrt(pred)
    print(f"    prediction A  : {pred:.2f} +/- {stat:.2f} (stat) +/- {syst:.2f} (syst from B/C/D stats)")
    if blind:
        print("    observed A    : BLINDED")
        return
    sig_tot = np.sqrt(pred + syst ** 2)
    ratio = A / pred if pred > 0 else np.nan
    pull = (A - pred) / sig_tot if sig_tot > 0 else np.nan
    print(f"    observed A    : {A:.0f}")
    print(f"    CLOSURE       : obs/pred = {ratio:.3f},  pull = {pull:+.2f} sigma")


def plot_plane(hh, xname, yname, xcut, ycut, x_sig_is_high, y_sig_is_high,
               title, path):
    vals = hh.values()
    xe = hh.axes[xname].edges
    ye = hh.axes[yname].edges
    fig, ax = plt.subplots(figsize=(9, 7))
    masked = np.ma.masked_where(vals.T <= 0, vals.T)
    pc = ax.pcolormesh(xe, ye, masked, norm=LogNorm())
    fig.colorbar(pc, ax=ax, label="Events")
    ax.axvline(xcut, color="red", ls="--")
    ax.axhline(ycut, color="red", ls="--")
    xlo, xhi = xe[0], xe[-1]
    ylo, yhi = ye[0], ye[-1]
    xsig_mid = 0.5 * (xcut + (xhi if x_sig_is_high else xlo))
    xcos_mid = 0.5 * (xcut + (xlo if x_sig_is_high else xhi))
    ysig_mid = 0.5 * (ycut + (yhi if y_sig_is_high else ylo))
    ycos_mid = 0.5 * (ycut + (ylo if y_sig_is_high else yhi))
    for lbl, xx, yy in (("A", xsig_mid, ysig_mid), ("B", xsig_mid, ycos_mid),
                        ("C", xcos_mid, ysig_mid), ("D", xcos_mid, ycos_mid)):
        ax.text(xx, yy, lbl, fontsize=22, color="red", ha="center", va="center",
                fontweight="bold")
    ax.set_xlabel(hh.axes[xname].label)
    ax.set_ylabel(hh.axes[yname].label)
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")


def closure_scan(h2, cat, scan_axis, cuts, fixed_cut, path, title):
    """Closure ratio obs/pred as a function of one region-boundary position."""
    ratios, errs, good_cuts = [], [], []
    for cut in cuts:
        if scan_axis == "dt":
            A, B, C, D = regions_from_2d(h2, {"cat": cat}, "cosa", "dt",
                                         fixed_cut, cut, True, True)
        else:
            A, B, C, D = regions_from_2d(h2, {"cat": cat}, "cosa", "dt",
                                         cut, fixed_cut, True, True)
        pred, syst = abcd_predict(B, C, D)
        if np.isnan(pred) or pred <= 0 or A <= 0:
            continue
        ratios.append(A / pred)
        errs.append(np.sqrt(pred + syst ** 2) / pred)
        good_cuts.append(cut)
    if not good_cuts:
        print(f"    scan {title}: no valid points, skipped")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(good_cuts, ratios, yerr=errs, fmt="o", capsize=3)
    ax.axhline(1.0, color="red", ls="--")
    nominal = DT_CUT if scan_axis == "dt" else COSA_CUT
    ax.axvline(nominal, color="gray", ls=":", label=f"nominal cut = {nominal}")
    ax.set_xlabel(f"{scan_axis} region boundary")
    ax.set_ylabel("observed / predicted in region A")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")


def intime_fraction_profile(h2, cats, ycut, path, rebin=20):
    """In-time fraction vs cosA. FLAT = independence assumption holds (eq. 1)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    any_pts = False
    for cat in cats:
        if cat not in list(h2.axes["cat"]):
            continue
        hh = h2[{"cat": cat}]
        vals = hh.values()
        yc = hh.axes["dt"].centers
        xe = hh.axes["cosa"].edges
        intime = yc > ycut
        nx = vals.shape[0] // rebin * rebin
        tot = vals[:nx].reshape(-1, rebin, vals.shape[1]).sum(axis=1)
        num = tot[:, intime].sum(axis=1)
        den = tot.sum(axis=1)
        xctr = 0.5 * (xe[:nx:rebin] + xe[rebin:nx + 1:rebin])
        ok = den > 0
        frac = np.where(ok, num / np.maximum(den, 1), np.nan)
        err = np.where(ok, np.sqrt(np.maximum(frac * (1 - frac), 1e-12) / np.maximum(den, 1)), np.nan)
        ax.errorbar(xctr[ok], frac[ok], yerr=err[ok], fmt="o", ms=3, label=cat)
        any_pts = True
    if not any_pts:
        plt.close(fig)
        return
    ax.axvline(COSA_CUT, color="red", ls="--", label=f"cosA cut = {COSA_CUT}")
    ax.set_xlabel(r"$\cos\alpha_{min}$")
    ax.set_ylabel(f"fraction with dt > {DT_CUT:.0f} ns (in-time)")
    ax.set_title("Independence check: in-time fraction vs cosA (flat = ABCD valid)")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")


def overlay_1d(specs, xlabel, title, path, log_y=True):
    """specs = [(values, edges, label), ...] — normalized step overlay."""
    fig, ax = plt.subplots(figsize=(8, 6))
    drawn = False
    for vals, edges, label in specs:
        tot = vals.sum()
        if tot == 0:
            continue
        ax.step(edges[:-1], vals / tot, where="post", label=f"{label} (n={int(tot)})")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of events")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")


if __name__ == "__main__":
    # ── samples: same preprocessed-pkl pattern as your other scripts.
    #    Missing files are skipped with a warning so you can run partial sets.
    #    ADD YOUR COLLISION-DATA PKL to make the (blinded) prediction. ──
    SAMPLES = {
        "cosmic": "scripts/samples/Run3_Summer22_chs_AK4PFCands_v21_DTTrigCalib/Cosmic_DTTrigCalib_preprocessed.pkl",
        "nobptx": "samples/Summer22_CHS_v21_Cosmic/NoBPTX_preprocessed.pkl",
        "signal": "samples/Signal_Samples/Stau_300_100mm_preprocessed.pkl",
        # "collision": "samples/<YOUR_COLLISION_DATASET>_preprocessed.pkl",
    }
    combined_runnable = {}
    for tag, path in SAMPLES.items():
        if not os.path.exists(path):
            print(f"WARNING: {tag} pkl not found ({path}) — skipping")
            continue
        with open(path, "rb") as f:
            combined_runnable.update(pickle.load(f))
        print(f"Loaded {tag}: {path}")
    if not combined_runnable:
        raise SystemExit("No samples found — fix the paths in SAMPLES.")

    print("Starting ABCD processor...")
    executor = processor.FuturesExecutor(workers=8)
    runner = processor.Runner(executor=executor, schema=PFNanoAODSchema,
                              chunksize=50_000, skipbadfiles=True)
    out = runner(combined_runnable, treename="Events",
                 processor_instance=ABCDProcessor())

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "abcd_raw_output.pkl"), "wb") as f:
        pickle.dump(out, f)

    CATS = [("cosmic", "CosmicMC"), ("nobptx", "NoBPTX"),
            ("data", "Collision"), ("signal", "Signal")]
    h2 = out["plane2mu"]
    h1 = out["plane1mu"]

    # ════════════════════ printed report ════════════════════
    print("\n" + "=" * 80)
    print(" ABCD COSMIC BACKGROUND STUDY")
    print(f"   dimuon plane : cosA cut {COSA_CUT}, dt cut {DT_CUT} ns (timeNDof > {T_NDOF})")
    print(f"   1-muon plane : t cut {T1MU_CUT} ns, |dz| cut {DZ_CUT} cm, per hemisphere")
    print("=" * 80)

    for p, cat in CATS:
        if out[f"n_total_{p}"] == 0:
            continue
        blind = (p == "data" and BLIND_COLLISION_A)
        print(f"\n{'-' * 80}\n {cat}\n{'-' * 80}")
        print(f"  events processed                        : {out[f'n_total_{p}']}")
        print(f"  events with a method-C candidate        : {out[f'n_cand_{p}']}")
        print(f"  ... exactly 1 DisMuon (after dedup)     : {out[f'n_1mu_{p}']}")
        print(f"      no-time (ndof<={T_NDOF}) upper / lower  : "
              f"{out[f'n_1mu_notime_up_{p}']} / {out[f'n_1mu_notime_lo_{p}']}   <- ABCD-blind bucket")
        print(f"  ... >=2 DisMuons                        : {out[f'n_ge2_{p}']}")
        print(f"      same-hemisphere (no dt possible)    : {out[f'n_2mu_same_hemi_{p}']}   <- ABCD-blind bucket")
        print(f"      ndof-fail (no valid dt)             : {out[f'n_2mu_ndof_fail_{p}']}   <- ABCD-blind bucket")
        print(f"      classified into the dimuon plane    : {out[f'n_2mu_classified_{p}']}")
        print(f"  shower-candidate events (surviving collinear pair): {out[f'n_shower_{p}']}")

        if cat in list(h2.axes["cat"]):
            A, B, C, D = regions_from_2d(h2, {"cat": cat}, "cosa", "dt",
                                         COSA_CUT, DT_CUT, True, True)
            print_abcd("DIMUON ABCD (cosA x dt)", A, B, C, D, blind=blind)
            vals = h2[{"cat": cat}].values()
            yc = h2.axes["dt"].centers
            n_pos = vals[:, yc > abs(DT_CUT)].sum()
            print(f"    diagnostic: events with dt > +{abs(DT_CUT):.0f} ns "
                  f"(counted in-time; upward/mismeasured?) : {n_pos:.0f}")

        for hemi in ("upper", "lower"):
            sel_ok = (cat in list(h1.axes["cat"])) and (hemi in list(h1.axes["hemi"]))
            if not sel_ok:
                continue
            A, B, C, D = regions_from_2d(h1, {"cat": cat, "hemi": hemi}, "t", "dz",
                                         T1MU_CUT, DZ_CUT, True, False)
            print_abcd(f"SINGLE-MUON ABCD ({hemi} hemisphere, t x |dz|)",
                       A, B, C, D, blind=blind)

    # signal contamination (guide checklist item 3)
    if "Signal" in list(h2.axes["cat"]) or "Signal" in list(h1.axes["cat"]):
        print(f"\n{'-' * 80}\n SIGNAL CONTAMINATION OF THE CONTROL REGIONS (checklist item 3)")
        print(" If B, C, or D hold a non-negligible signal fraction, the prediction is")
        print(" biased — quote these numbers next to the background estimate.")
        print(f"{'-' * 80}")

    # ════════════════════ plots ════════════════════
    print("\nPlots:")
    for p, cat in CATS:
        blind = (p == "data" and BLIND_COLLISION_A)
        if cat in list(h2.axes["cat"]):
            plot_plane(h2[{"cat": cat}], "cosa", "dt", COSA_CUT, DT_CUT, True, True,
                       f"Dimuon ABCD plane — {cat}" + (" (A blinded in the report)" if blind else ""),
                       os.path.join(OUT_DIR, f"plane2mu_{cat}.pdf"))
            closure_scan(h2, cat, "dt", np.arange(-40.0, -9.0, 1.0), COSA_CUT,
                         os.path.join(OUT_DIR, f"scan_dtcut_{cat}.pdf"),
                         f"Closure vs dt boundary — {cat}")
            closure_scan(h2, cat, "cosa", np.arange(-0.995, -0.899, 0.005), DT_CUT,
                         os.path.join(OUT_DIR, f"scan_cosacut_{cat}.pdf"),
                         f"Closure vs cosA boundary — {cat}")
            hh = out["dphi_vs_cosa"][{"cat": cat}]
            fig, ax = plt.subplots(figsize=(9, 7))
            masked = np.ma.masked_where(hh.values().T <= 0, hh.values().T)
            pc = ax.pcolormesh(hh.axes["cosa"].edges, hh.axes["dphi"].edges,
                               masked, norm=LogNorm())
            fig.colorbar(pc, ax=ax, label="Events")
            ax.set_xlabel(r"$\cos\alpha_{min}$")
            ax.set_ylabel(r"$|\Delta\phi|$")
            ax.set_title(f"Beam-halo check — {cat} (halo: dphi~0, cosA -0.6..-0.95; cosmic: dphi~pi, cosA~-1)")
            fig.savefig(os.path.join(OUT_DIR, f"dphi_vs_cosa_{cat}.pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"    saved: {os.path.join(OUT_DIR, f'dphi_vs_cosa_{cat}.pdf')}")
        for hemi in ("upper", "lower"):
            if cat in list(h1.axes["cat"]) and hemi in list(h1.axes["hemi"]):
                plot_plane(h1[{"cat": cat, "hemi": hemi}], "t", "dz",
                           T1MU_CUT, DZ_CUT, True, False,
                           f"Single-muon ABCD plane ({hemi}) — {cat}",
                           os.path.join(OUT_DIR, f"plane1mu_{hemi}_{cat}.pdf"))

    intime_fraction_profile(h2, [c for _, c in CATS],
                            DT_CUT, os.path.join(OUT_DIR, "intime_frac_vs_cosa.pdf"))

    # MET: 1mu vs >=2mu per dataset (the anti-selection check)
    hmet = out["met"]
    for p, cat in CATS:
        if cat not in list(hmet.axes["cat"]):
            continue
        specs = []
        for kind in ("1mu", "ge2mu"):
            if kind in list(hmet.axes["kind"]):
                hh = hmet[{"cat": cat, "kind": kind}]
                specs.append((hh.values(), hh.axes["val"].edges, kind))
        overlay_1d(specs, "PFMET [GeV]",
                   f"PFMET, 1mu vs >=2mu — {cat} (vertical line = SR cut)",
                   os.path.join(OUT_DIR, f"met_1mu_vs_2mu_{cat}.pdf"))

    # shower multiplicity overlays
    hm = out["mult"]
    for p, cat in CATS:
        if cat not in list(hm.axes["cat"]):
            continue
        specs = []
        for kind in ("all", "shower"):
            if kind in list(hm.axes["kind"]):
                hh = hm[{"cat": cat, "kind": kind}]
                specs.append((hh.values(), hh.axes["val"].edges, kind))
        overlay_1d(specs, "DisMuons per event",
                   f"Muon multiplicity, all >=2mu vs shower-candidate — {cat}",
                   os.path.join(OUT_DIR, f"shower_mult_{cat}.pdf"))

    # ════════════════════ shower dumps (Fireworks pick lists) ════════════════
    for p, cat in CATS:
        rows = out[f"shower_rows_{p}"]
        if not rows:
            continue
        path = os.path.join(OUT_DIR, f"shower_candidates_{cat}.txt")
        with open(path, "w") as fh:
            fh.write(f"{cat}: collinear (cosA > {SHOWER_COSA}, non-duplicate) pairs that SURVIVE\n")
            fh.write("the outertrack + segment cos(angle) < -0.8 tests — cosmic-shower candidates.\n")
            fh.write("tspread = max-min timeAtIpInOut among muons with timeNDof > "
                     f"{T_NDOF} (small = coherent, shower-like)\n")
            fh.write("dt = hemisphere-leader dt (nan = same hemisphere / no valid time)\n\n")
            fh.write("{:>8} {:>7} {:>13} {:>4} {:>8} {:>8} {:>8} {:>6} {:>9} {:>9}\n".format(
                "run", "lumi", "event", "nMu", "cosA", "cos_out", "cos_seg",
                "nSeg", "tspread", "dt"))
            for (rr, ll, ee, nm, cab, co, csg, ns, ts, dtv) in rows:
                fh.write("{:>8d} {:>7d} {:>13d} {:>4d} {:>8.3f} {:>8.3f} {:>8.3f} "
                         "{:>6d} {:>9.2f} {:>9.2f}\n".format(rr, ll, ee, nm, cab, co, csg, ns, ts, dtv))
        pick = os.path.join(OUT_DIR, f"shower_pick_{cat}.txt")
        seen = set()
        with open(pick, "w") as fh:
            for (rr, ll, ee, *_rest) in rows:
                key = (rr, ll, ee)
                if key in seen:
                    continue
                seen.add(key)
                fh.write(f"{rr}:{ll}:{ee}\n")
        print(f"\n    shower dump : {path}  ({len(rows)} pairs)")
        print(f"    pick list   : {pick}  ({len(seen)} events, run:lumi:event — feed to edmPickEvents.py)")

    print("\nDone. Read the report top to bottom; then look at plane2mu_*.pdf, "
          "intime_frac_vs_cosa.pdf, and the scan_* plots before trusting any number.")
