import pickle
import math
import numpy as np
import awkward as ak
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import coffea.nanoevents.methods.vector as vector

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

def _p3(pt, eta, phi):
    return (pt*math.cos(phi), pt*math.sin(phi), pt*math.sinh(eta), pt*math.cosh(eta))

def _cosA(m1, m2):
    px1, py1, pz1, p1 = _p3(m1[0], m1[1], m1[2])
    px2, py2, pz2, p2 = _p3(m2[0], m2[1], m2[2])
    denom = p1 * p2
    return (px1*px2 + py1*py2 + pz1*pz2) / denom if denom else -1000.0

class CosAStudyProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {}
        for p in ("cosmic", "nobptx"):
            self.output[f"n_cosA_study_events_{p}"] = 0
            self.output[f"n_cosA_study_muons_in_{p}"] = 0
            self.output[f"n_cosA_study_pairs_{p}"] = 0
            self.output[f"n_cosA_study_events_flagged_{p}"] = 0
            self.output[f"n_cosA_collinear_events_flagged_{p}"] = 0
            self.output[f"n_cosA_study_events_surviving_{p}"] = 0
            self.output[f"n_cosA_study_muons_surviving_{p}"] = 0
            self.output[f"n_cosA_dt_study_events_vetoed_{p}"] = 0
            self.output[f"n_cosA_dt_study_events_surviving_{p}"] = 0
            self.output[f"n_cosA_dt_study_muons_surviving_{p}"] = 0
            self.output[f"surviving_event_ids_{p}"] = []
            self.output[f"surviving_event_muons_{p}"] = []
            self.output[f"n_events_multi_dimuon_predup_{p}"] = 0
            self.output[f"n_events_multi_dimuon_postdup_{p}"] = 0
            self.output[f"collinear_event_ids_{p}"] = []

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")

        is_cosmic = "Cosmic" in dataset or dataset.startswith("LooseMu") or dataset == "test_cosmics_calib"
        is_nobptx = "NoBPTX" in dataset

        # This standalone only handles cosmic / nobptx
        if not (is_cosmic or is_nobptx):
            return self.output

        prefix_cs = "cosmic" if is_cosmic else "nobptx"

        # ── Minimal DisMuon zip (only the fields this study needs) ──
        events["DisMuon"] = ak.zip(
            {
                "pt":             events.DisMuon.pt,
                "eta":            events.DisMuon.eta,
                "phi":            events.DisMuon.phi,
                "mass":           events.DisMuon.mass,
                "charge":         events.DisMuon.charge,
                "timeNDof":       events.DisMuon.timeNDof,
                "timeAtIpInOut":  events.DisMuon.timeAtIpInOut,
                "mediumId":       events.DisMuon.mediumId,
                "pfRelIso03_all": events.DisMuon.pfRelIso03_all,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # ── Kinematic cuts ──
        dismuon_mask = (events.DisMuon.pt > 30) & (abs(events.DisMuon.eta) < 2.4)
        events["DisMuon"] = events.DisMuon[dismuon_mask]

        # ── NoBPTX: remove events with good vertices ──
        if is_nobptx:
            events = events[events.PV.npvsGood == 0]

        evid = ak.zip({
            "run":   events.run,
            "lumi":  events.luminosityBlock,
            "event": events.event,
        })

        dis_muons = events.DisMuon
        n_dismuons = ak.num(dis_muons)

        # ── GLOBAL DUPLICATE TRACK REMOVAL ──
        mask_has_muons_g = (n_dismuons >= 1)
        if ak.sum(mask_has_muons_g) == 0:
            return self.output
        dis_muons = dis_muons[mask_has_muons_g]
        evid = evid[mask_has_muons_g]

        sorted_all = dis_muons[ak.argsort(dis_muons.pt, axis=1, ascending=False)]
        lead_all = sorted_all[:, 0]
        lead_quality_mask_g = (lead_all.mediumId == True) & (lead_all.pfRelIso03_all < 0.18)
        sorted_all = sorted_all[lead_quality_mask_g]
        evid = evid[lead_quality_mask_g]
        if len(sorted_all) == 0:
            return self.output

        local_i_g = ak.local_index(sorted_all, axis=1)
        a_g, b_g   = ak.unzip(ak.cartesian([sorted_all, sorted_all], axis=1, nested=True))
        ia_g, ib_g = ak.unzip(ak.cartesian([local_i_g, local_i_g], axis=1, nested=True))

        deta_g = a_g.eta - b_g.eta
        dphi_g = a_g.delta_phi(b_g)
        dpt_g  = a_g.pt - b_g.pt
        mask_sc_g = (a_g.charge * b_g.charge) > 0

        # a_g is a duplicate if it matches an earlier (higher-pT) muon b_g
        match_g = (
            mask_sc_g
            & (abs(deta_g) < 0.01)
            & (abs(dphi_g) < 0.001)
            & (abs(dpt_g)  < 0.5)
            & (ib_g < ia_g)
        )
        is_duplicate_g = ak.any(match_g, axis=2)
        dis_muons = sorted_all[~is_duplicate_g]
        self.output[f"n_events_multi_dimuon_predup_{prefix_cs}"] += int(ak.sum(ak.num(sorted_all) > 1))
        self.output[f"n_events_multi_dimuon_postdup_{prefix_cs}"] += int(ak.sum(ak.num(dis_muons) > 1))

        # ==========================================
        # cosA COSMIC REMOVAL STUDY
        # ==========================================
        n_cs_all = ak.num(dis_muons)
        cs_mask  = (n_cs_all >= 2)
        if ak.sum(cs_mask) == 0:
            return self.output
        cs_muons = dis_muons[cs_mask]
        evid = evid[cs_mask]

        # Sort by pT; apply mediumId + iso to the lead only
        cs_sorted = cs_muons[ak.argsort(cs_muons.pt, axis=1, ascending=False)]
        cs_lead = cs_sorted[:, 0]
        cs_lead_pass = (cs_lead.mediumId == True) & (cs_lead.pfRelIso03_all < 0.18)
        cs_sorted = cs_sorted[cs_lead_pass]
        evid = evid[cs_lead_pass]
        if len(cs_sorted) == 0:
            return self.output

        cs_lead = cs_sorted[:, 0]
        cs_sub  = cs_sorted[:, 1:]

        self.output[f"n_cosA_study_events_{prefix_cs}"] += len(cs_sorted)
        self.output[f"n_cosA_study_muons_in_{prefix_cs}"] += int(ak.sum(ak.num(cs_sorted)))

        # cosA between lead and each sub-leading muon
        dot_cs   = cs_lead.px * cs_sub.px + cs_lead.py * cs_sub.py + cs_lead.pz * cs_sub.pz
        denom_cs = cs_lead.p * cs_sub.p
        cosA_cs  = ak.where(denom_cs != 0, dot_cs / denom_cs, -1000.0)

        # Veto entire event if ANY pair is back-to-back (cosA < -0.99)
        event_has_cosmic = ak.any(cosA_cs < -0.99, axis=1)
        # Veto entire event if ANY pair is collinear (cosA > 0.99)
        event_has_collinear = ak.any(cosA_cs > 0.99, axis=1)

        survive_b2b       = ~event_has_cosmic
        collinear_removed = survive_b2b & event_has_collinear   # incremental removal
        survive_cosA      = survive_b2b & ~event_has_collinear

        self.output[f"n_cosA_study_pairs_{prefix_cs}"] += int(ak.sum(ak.num(cs_sub)))
        self.output[f"n_cosA_study_events_flagged_{prefix_cs}"] += int(ak.sum(event_has_cosmic))
        self.output[f"n_cosA_collinear_events_flagged_{prefix_cs}"] += int(ak.sum(collinear_removed))
        evid_collinear = evid[collinear_removed]
        self.output[f"collinear_event_ids_{prefix_cs}"] += [
            (d["run"], d["lumi"], d["event"]) for d in ak.to_list(evid_collinear)]
        self.output[f"n_cosA_study_events_surviving_{prefix_cs}"] += int(ak.sum(survive_cosA))

        surviving_muons = cs_sorted[survive_cosA]
        evid_surv = evid[survive_cosA]

        self.output[f"n_cosA_study_muons_surviving_{prefix_cs}"] += int(ak.sum(ak.num(surviving_muons)))
        if len(surviving_muons) == 0:
            return self.output

        # ── Δt for surviving events: upper - lower hemisphere ──
        # Require timeNDof > 7 ONLY on the muons that enter the timing cut
        surv_upper_all = surviving_muons[surviving_muons.phi > 0]
        surv_lower_all = surviving_muons[surviving_muons.phi < 0]
        surv_has_both  = (ak.num(surv_upper_all) >= 1) & (ak.num(surv_lower_all) >= 1)

        if ak.sum(surv_has_both) > 0:
            su = surv_upper_all[surv_has_both]
            sl = surv_lower_all[surv_has_both]
            surv_upper = su[ak.argsort(su.pt, axis=1, ascending=False)][:, 0]
            surv_lower = sl[ak.argsort(sl.pt, axis=1, ascending=False)][:, 0]

            # BOTH the upper and lower lead must have valid timing (ndof > 7).
            # If EITHER fails, the dt comparison isn't trustworthy → veto the event.
            ndof_ok = (surv_upper.timeNDof > 7) & (surv_lower.timeNDof > 7)

            surv_dt = surv_upper.timeAtIpInOut - surv_lower.timeAtIpInOut
            dt_pass = ndof_ok & (surv_dt >= -20)
            dt_fail = ~dt_pass

            self.output[f"n_cosA_dt_study_events_vetoed_{prefix_cs}"] += int(ak.sum(dt_fail))
            self.output[f"n_cosA_dt_study_events_surviving_{prefix_cs}"] += int(ak.sum(dt_pass))

            final_surviving = surviving_muons[surv_has_both][dt_pass]
            evid_final = evid_surv[surv_has_both][dt_pass]
            self.output[f"surviving_event_ids_{prefix_cs}"] += [
                (d["run"], d["lumi"], d["event"]) for d in ak.to_list(evid_final)]
            self.output[f"surviving_event_muons_{prefix_cs}"] += [
                [(m["pt"], m["eta"], m["phi"], m["timeAtIpInOut"]) for m in ev]
                for ev in ak.to_list(final_surviving)]
            self.output[f"n_cosA_dt_study_muons_surviving_{prefix_cs}"] += int(ak.sum(ak.num(final_surviving)))

        # Events surviving cosA but with no opposite-hemisphere pair: dt cut can't apply
        surv_no_both = surviving_muons[~surv_has_both]
        if len(surv_no_both) > 0:
            self.output[f"n_cosA_dt_study_events_surviving_{prefix_cs}"] += len(surv_no_both)
            self.output[f"n_cosA_dt_study_muons_surviving_{prefix_cs}"] += int(ak.sum(ak.num(surv_no_both)))
            evid_no_both = evid_surv[~surv_has_both]
            self.output[f"surviving_event_ids_{prefix_cs}"] += [
                (d["run"], d["lumi"], d["event"]) for d in ak.to_list(evid_no_both)]
            self.output[f"surviving_event_muons_{prefix_cs}"] += [
                [(m["pt"], m["eta"], m["phi"], m["timeAtIpInOut"]) for m in ev]
                for ev in ak.to_list(surv_no_both)]

        return self.output

    def postprocess(self, accumulator):
        return accumulator

class CosmicCalibLookupProcessor(processor.ProcessorABC):
    """Look up a fixed set of (run, lumi, event) and report their DisMuons.
    Only the baseline pt>30 / |eta|<2.4 muon definition is applied — NO
    duplicate removal and NO cosA/timing cuts — so the printed muons are the
    same physical muons, just with cosmic calibration applied."""
    def __init__(self, target_ids):
        self.target_ids = set(target_ids)

    def process(self, events):
        run  = ak.to_numpy(events.run)
        lumi = ak.to_numpy(events.luminosityBlock)
        evno = ak.to_numpy(events.event)

        mask = np.zeros(len(events), dtype=bool)
        for (r, l, e) in self.target_ids:
            mask |= (run == r) & (lumi == l) & (evno == e)
        if not mask.any():
            return {}

        sel      = events[mask]
        sel_run  = run[mask]
        sel_lumi = lumi[mask]
        sel_evno = evno[mask]

        kin = (sel.DisMuon.pt > 30) & (abs(sel.DisMuon.eta) < 2.4)
        mu = ak.zip({
            "pt":            sel.DisMuon.pt,
            "eta":           sel.DisMuon.eta,
            "phi":           sel.DisMuon.phi,
            "timeAtIpInOut": sel.DisMuon.timeAtIpInOut,
        })[kin]
        mu = mu[ak.argsort(mu.pt, axis=1, ascending=False)]
        mu_list = ak.to_list(mu)

        out = {}
        for i in range(len(sel_run)):
            key = (int(sel_run[i]), int(sel_lumi[i]), int(sel_evno[i]))
            out[key] = [
                (float(m["pt"]), float(m["eta"]), float(m["phi"]), float(m["timeAtIpInOut"]))
                for m in mu_list[i]
            ]
        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == "__main__":
    cosmic_pkl = "scripts/samples/Summer22_CHS_collisionCalib_v19_Cosmic/Cosmic_CollisionCalib_preprocessed.pkl"
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
        processor_instance=CosAStudyProcessor(),
    )

    # ── Write every collinear (cosA > +0.99) removed event for the COLLISION-CALIB MC ──
    collinear_ids = out["collinear_event_ids_cosmic"]
    collinear_txt = "collinear_cosA_gt_0p99_collisionCalib_v19_Cosmic.txt"
    with open(collinear_txt, "w") as f:
        for (run, lumi, event) in collinear_ids:
            f.write(f"{run}:{lumi}:{event}\n")
    print(f"Wrote {len(collinear_ids)} collinear (cosA > +0.99) removed events to {collinear_txt}")

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
        n_collinear = out[f'n_cosA_collinear_events_flagged_{prefix}']
        n_surv    = out[f'n_cosA_study_events_surviving_{prefix}']
        n_mu_surv = out[f'n_cosA_study_muons_surviving_{prefix}']

        n_dt_vetoed  = out[f'n_cosA_dt_study_events_vetoed_{prefix}']
        n_dt_surv    = out[f'n_cosA_dt_study_events_surviving_{prefix}']
        n_dt_mu_surv = out[f'n_cosA_dt_study_muons_surviving_{prefix}']

        print(f"\n  {ds_name}:")
        print(f"    Events entering study (>=2 muons, lead passes cuts): {n_evt}")
        print(f"    Total muons entering:                                {n_mu_in}")
        print(f"    Lead vs sub-leading pairs checked:                   {n_pairs}")
        print(f"    ── After cosA cut (cosA < -0.99 OR cosA > +0.99 → veto event) ──")
        print(f"    Events vetoed by cosA back-to-back (< -0.99):        {n_flagged}")
        print(f"    Events vetoed by cosA collinear   (> +0.99):         {n_collinear}")
        print(f"    Events surviving both cosA cuts:                     {n_surv}")
        if n_evt > 0:
            print(f"    cosA survival rate: {n_surv}/{n_evt} = {(n_surv/n_evt)*100:.2f}%")
        print(f"    Muons in cosA-surviving events:                      {n_mu_surv}")
        
        print(f"    ── After dt cut (dt < -20 ns → veto event) ──")
        print(f"    Events vetoed by dt:                                 {n_dt_vetoed}")
        print(f"    Events surviving cosA + dt:                          {n_dt_surv}")
        if n_evt > 0:
            print(f"    Combined survival rate: {n_dt_surv}/{n_evt} = {(n_dt_surv/n_evt)*100:.2f}%")
            print(f"    Combined veto rate:     {(n_flagged + n_collinear + n_dt_vetoed)}/{n_evt} = {((n_flagged + n_collinear + n_dt_vetoed)/n_evt)*100:.2f}%")
        print(f"    Muons in final surviving events:                     {n_dt_mu_surv}")
        if n_mu_in > 0:
            print(f"    Final muon survival rate: {n_dt_mu_surv}/{n_mu_in} = {(n_dt_mu_surv/n_mu_in)*100:.2f}%")
        ids = out[f"surviving_event_ids_{prefix}"]
        muons_per_event = out[f"surviving_event_muons_{prefix}"]
        print(f"    ── First 10 surviving events (Run:Lumi:Event) for fireworks ──")
        for (run, lumi, event), muons in list(zip(ids, muons_per_event))[:10]:
            print(f"      {run}:{lumi}:{event}")
            for pt, eta, phi, t in muons:
                print(f"        pt={pt:8.2f}  eta={eta:+.4f}  phi={phi:+.4f}  timeAtIpInOut={t:+.3f}")
            lead = muons[0]
            for k in range(1, len(muons)):
                print(f"        cosA(lead, mu{k}) = {_cosA(lead, muons[k]):+.5f}")

        n_pre  = out[f'n_events_multi_dimuon_predup_{prefix}']
        n_post = out[f'n_events_multi_dimuon_postdup_{prefix}']
        print(f"    ── Duplicate-removal effect (events with >1 DisMuon) ──")
        print(f"    Events with >1 DisMuon before duplicate removal: {n_pre}")
        print(f"    Events with >1 DisMuon after  duplicate removal: {n_post}")
        print(f"    Events reduced to <=1 muon by duplicate removal: {n_pre - n_post}")
    print("="*80 + "\n")

    # =====================================================================
    #  COSMIC-CALIB LOOKUP: same 10 events, same muons, for comparison
    # =====================================================================
    cc_ids        = out["surviving_event_ids_cosmic"][:10]
    cc_muons      = out["surviving_event_muons_cosmic"][:10]
    cc_target_ids = [(int(r), int(l), int(e)) for (r, l, e) in cc_ids]

    if not cc_target_ids:
        print("No surviving cosmic events to look up; skipping cosmic-calib comparison.")
    else:
        cosmiccalib_pkl = "scripts/samples/Summer22_CHS_v19_Cosmic/Cosmic_CosmicCalib_preprocessed.pkl"
        print(f"\nLoading Cosmic-calibration sample from {cosmiccalib_pkl} for comparison...")
        with open(cosmiccalib_pkl, "rb") as f:
            cosmiccalib_runnable = pickle.load(f)

        cc_lookup = runner(
            cosmiccalib_runnable,
            treename="Events",
            processor_instance=CosmicCalibLookupProcessor(cc_target_ids),
        )
        '''
        print("="*80)
        print(" COLLISION-CALIB vs COSMIC-CALIB  (same 10 events, same muons)")
        print(" (baseline pt>30 / |eta|<2.4 only — no dedup, no cosA/timing cuts)")
        print("="*80)
        for (run, lumi, event), coll_muons in zip(cc_ids, cc_muons):
            key = (int(run), int(lumi), int(event))
            print(f"\n  {run}:{lumi}:{event}")

            print("    ── collision calib ──")
            for pt, eta, phi, t in coll_muons:
                print(f"      pt={pt:8.2f}  eta={eta:+.4f}  phi={phi:+.4f}  timeAtIpInOut={t:+.3f}")
            for k in range(1, len(coll_muons)):
                print(f"      cosA(lead, mu{k}) = {_cosA(coll_muons[0], coll_muons[k]):+.5f}")

            print("    ── cosmic calib ──")
            cc = cc_lookup.get(key)
            if cc is None:
                print("      (event not found in cosmic-calib sample)")
            elif len(cc) == 0:
                print("      (no DisMuons pass pt>30 / |eta|<2.4 here)")
            else:
                for pt, eta, phi, t in cc:
                    print(f"      pt={pt:8.2f}  eta={eta:+.4f}  phi={phi:+.4f}  timeAtIpInOut={t:+.3f}")
                for k in range(1, len(cc)):
                    print(f"      cosA(lead, mu{k}) = {_cosA(cc[0], cc[k]):+.5f}")
        print("="*80)
        '''
    print("Done!")