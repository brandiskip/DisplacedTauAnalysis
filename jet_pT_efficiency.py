import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import hist
import vector
from hist import Hist, axis, intervals
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import coffea.nanoevents.methods
import json
np.set_printoptions(precision=6, suppress=False, threshold=np.inf)

def get_ratio_histogram(passing_probes, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = passing_probes.values(flow=True) / denominator.values(flow=True)
                                 
    ratio = hist.Hist(hist.Hist(*passing_probes.axes))
    ratio[:] = np.nan_to_num(ratio_values)
    yerr = intervals.ratio_uncertainty(passing_probes.values(), denominator.values(), uncertainty_type="efficiency")
    return ratio, yerr

def plot_efficiency(passing_probes, denominator, log=False, **kwargs):
    ratio_hist, yerr = get_ratio_histogram(passing_probes, denominator)
    plt.ylabel('efficiency')
    if log:  plt.xscale('log')
    return ratio_hist.plot1d(histtype="errorbar", yerr=yerr, xerr=True, flow="none", **kwargs)

# --- JSON helpers (minimal, append-safe) ---
def _hist_to_payload(Hnum: Hist, Hden: Hist):
    edges = np.asarray(Hnum.axes[0].edges)  # same for den
    return {
        "edges": edges.tolist(),
        "num": Hnum.values(flow=False).tolist(),
        "den": Hden.values(flow=False).tolist(),
    }

def _append_histograms_to_json(json_path, sample_name, category, payload_dict):
    """
    payload_dict = {"pt": {...}, "Lxy": {...}, "eta": {...}}, where each {...} comes from _hist_to_payload
    """
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            store = json.load(f)
    else:
        store = {}
    store.setdefault(sample_name, {})
    store[sample_name][category] = payload_dict
    with open(json_path, "w") as f:
        json.dump(store, f, indent=2)

def _payload_to_hists(payload, axis_name):
    """Rebuild Hist objects from JSON so you can reuse plot_efficiency() unchanged."""
    edges = np.asarray(payload["edges"])
    A = axis.Variable(edges, name=axis_name)
    Hden = Hist(A); Hnum = Hist(A)

    den = np.asarray(payload["den"], dtype=float)
    num = np.asarray(payload["num"], dtype=float)

    # Guard against zero denominators without changing your structure:
    # bins with den==0 get num=0, den=1 so ratio=0 and uncertainties are well-defined.
    z = den == 0
    if np.any(z):
        den = den.copy(); num = num.copy()
        den[z] = 1.0
        num[z] = 0.0

    Hden[:] = den
    Hnum[:] = num
    return Hnum, Hden

def _sanitize_inplace(Hnum: Hist, Hden: Hist):
    den = Hden.values(flow=False)
    num = Hnum.values(flow=False)
    z = den == 0
    if np.any(z):
        den = den.copy(); num = num.copy()
        den[z] = 1.0   # keep ratio defined
        num[z] = 0.0
        Hden[:] = den
        Hnum[:] = num

json_cache = "efficiency_cache.json"

# Load the file
filenames = {
    #'Stau_100_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_1mm_no_cut'    : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_10mm_no_cut'   : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_100mm_no_cut'  : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_1mm_no_cut'    : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_10mm_no_cut'   : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_100mm_no_cut'  : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1mm_no_cut'    : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_10mm_no_cut'   : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_100mm_no_cut'  : 'root://cmseos.fnal.gov///store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
}

PFNanoAODSchema.mixins["DisMuon"] = "Muon"
samples = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array: 
    mval = first.metric_table(second) 
    return ak.all(mval > threshold, axis=-1)

pT_output_dir = "jets_pT_efficiency"
Lxy_output_dir = "jets_Lxy_efficiency"
eta_output_dir = "jets_eta_efficiency"
d0_output_dir = "jets_d0_efficiency"

os.makedirs(pT_output_dir, exist_ok=True)
os.makedirs(Lxy_output_dir, exist_ok=True)
os.makedirs(eta_output_dir, exist_ok=True)
os.makedirs(d0_output_dir, exist_ok=True)

if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        # add dxy to jet fields
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = abs(ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, \
                ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)))
        events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")
        dxy_err = abs(ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0Err, axis = 2))
        events['Jet'] = ak.with_field(events.Jet, dxy_err, where="dxy_err")
        vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.parent.vx
        vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.parent.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        parent_with_Lxy = ak.with_field(events.GenVisTau.parent, Lxy, where="Lxy")
        events['GenVisTau'] = ak.with_field(events.GenVisTau, parent_with_Lxy, where="parent")

        noise_mask = (
                     (events.Flag.goodVertices == 1) 
                     & (events.Flag.globalSuperTightHalo2016Filter == 1)
                     & (events.Flag.EcalDeadCellTriggerPrimitiveFilter == 1)
                     & (events.Flag.BadPFMuonFilter == 1)
                     & (events.Flag.BadPFMuonDzFilter == 1)
                     & (events.Flag.hfNoisyHitsFilter == 1)
                     & (events.Flag.eeBadScFilter == 1)
                     & (events.Flag.ecalBadCalibFilter == 1)
                         )

        trigger_mask = (
            events.HLT.PFMET120_PFMHT120_IDTight
            | events.HLT.PFMET130_PFMHT130_IDTight
            | events.HLT.PFMET140_PFMHT140_IDTight
            | events.HLT.PFMETNoMu120_PFMHTNoMu120_IDTight
            | events.HLT.PFMETNoMu130_PFMHTNoMu130_IDTight
            | events.HLT.PFMETNoMu140_PFMHTNoMu140_IDTight
            | events.HLT.PFMET120_PFMHT120_IDTight_PFHT60
            | events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF
            | events.HLT.PFMETTypeOne140_PFMHT140_IDTight
            | events.HLT.MET105_IsoTrk50
            | events.HLT.MET120_IsoTrk50
        )

        events = events[noise_mask & trigger_mask]

        ## find staus and their tau children
        gpart = events.GenPart
        events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] 

        events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                          (events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                        (events.staus.distinctChildren.hasFlags("fromHardProcess")) \
                                                         ]
        events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & \
                                                        (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & \
                                                        (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & \
                                                        (events.GenVisTau.parent.hasFlags("fromHardProcess")) & \
                                                        (events.GenVisTau.parent.Lxy < 100.0) & \
                                                        (events.GenVisTau.pt > 20) & \
                                                        (abs(events.GenVisTau.eta) < 2.4)]

        d0 = abs((events.GenVisStauTaus.parent.vy - events.GenVtx.y) * np.cos(events.GenVisStauTaus.parent.phi) - \
              (events.GenVisStauTaus.parent.vx - events.GenVtx.x) * np.sin(events.GenVisStauTaus.parent.phi))
        events['GenVisStauTaus'] = ak.with_field(events.GenVisStauTaus, d0, where="d0")

        events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
        staus_taus = events['staus_taus']

        mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        mask_tauh = ~mask_taul

        one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
        one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

        filtered_events = events[one_tauh_evt & one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau
    
        tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus_taus.eta) < 2.4), axis=-1)
        num_taus = ak.num(filtered_events.staus_taus[tau_selections])
        num_tau_mask = num_taus > 1
        cut_filtered_events = filtered_events[num_tau_mask]
        
        '''
        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
        jets_neHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.neHEF < 0.99)]
        jets_neEmEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.neEmEF < 0.9)]
        jets_ch_ne_Mult = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & ((cut_filtered_events.Jet.chMultiplicity + cut_filtered_events.Jet.neMultiplicity) > 1)]
        jets_chHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.chHEF > 0.01)]
        jets_chMultiplicity = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.chMultiplicity > 0)]

        jet_matched_gen_vis_taus = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus = ak.drop_none(jet_matched_gen_vis_taus)

        jet_matched_gen_vis_taus_neHEF = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_neHEF = ak.drop_none(jet_matched_gen_vis_taus_neHEF)

        jet_matched_gen_vis_taus_neEmEF = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_neEmEF = ak.drop_none(jet_matched_gen_vis_taus_neEmEF)

        jet_matched_gen_vis_taus_ch_ne_Mult = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_ch_ne_Mult = ak.drop_none(jet_matched_gen_vis_taus_ch_ne_Mult)

        jet_matched_gen_vis_taus_chHEF = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_chHEF = ak.drop_none(jet_matched_gen_vis_taus_chHEF)

        jet_matched_gen_vis_taus_chMultiplicity = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_chMultiplicity = ak.drop_none(jet_matched_gen_vis_taus_chHEF)

        jets_variants = {
            "baseline (pt>20,|eta|<2.4)": jets,
            "neHEF < 0.99":               jets_neHEF,
            "neEmEF < 0.90":              jets_neEmEF,
            "ch+ne mult > 1":             jets_ch_ne_Mult,
            "chHEF > 0.01":               jets_chHEF,
            "chMultiplicity > 0":         jets_chMultiplicity,
        }
        
        # η denominator
        eta_bins = np.arange(-2.1, 2.1 + 1e-6, 0.1)
        eta_axis = axis.Variable(eta_bins, name="eta")
        hist_eta_den = Hist(eta_axis)
        hist_eta_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.eta, axis=None).compute())
        
        # Lxy denominator 
        Lxy_bins = np.arange(0, 100 + 1e-6, 5.0)
        Lxy_axis = axis.Variable(Lxy_bins, name="Lxy")
        hist_Lxy_den = Hist(Lxy_axis)
        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy, axis=None).compute())
        
        # pT denominator 
        pt_bins_low = np.arange(20, 101, 20)
        pt_bins_med = np.arange(100, 400, 30)
        pt_bins_high = np.arange(400, 600, 40)
        pt_bins_higher = np.arange(600, 1000, 50)
        pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))
        pt_axis = axis.Variable(pt_bins_eff, flow=False, name="GenVisTau_pt")
        hist_pt_den = Hist(pt_axis)
        hist_pt_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None).compute())
        
        # ---------------------------
        # Overlay efficiency vs η
        # ---------------------------
        plt.clf()
        for label, jcol in jets_variants.items():
            # Match GEN to jets for THIS variant (and drop events with no match)
            matched_gen = jcol.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
            matched_gen = ak.drop_none(matched_gen)

            # Numerator: use GEN variables of the matched GenVisStauTaus
            hist_eta_num = Hist(eta_axis)
            if ak.any(~ak.is_none(matched_gen)):
                hist_eta_num.fill(ak.flatten(matched_gen.eta, axis=None).compute())

            # Plot this curve on the same axes
            plot_efficiency(hist_eta_num, hist_eta_den, label=label)

        plt.ylim(top=1.05)
        plt.xlabel(r"$\eta$ (GenVisStauTau)")
        plt.title(f"Matching efficiency vs η (all jet variants) — {sample_name}")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eta_output_dir, f"eff_vs_eta_overlay_{sample_name}.pdf"))
        
        # ---------------------------
        # Overlay efficiency vs Lxy
        # ---------------------------
        plt.clf()
        for label, jcol in jets_variants.items():
            matched_gen = jcol.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
            matched_gen = ak.drop_none(matched_gen)

            hist_Lxy_num = Hist(Lxy_axis)
            if ak.any(~ak.is_none(matched_gen)):
                hist_Lxy_num.fill(ak.flatten(matched_gen.parent.Lxy, axis=None).compute())

            plot_efficiency(hist_Lxy_num, hist_Lxy_den, label=label)

        plt.ylim(top=1.05)
        plt.xlabel(r"$L_{xy}$ (cm) of GenVisStauTau parent")
        plt.title(f"Matching efficiency vs $L_{{xy}}$ (all jet variants) — {sample_name}")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Lxy_output_dir, f"eff_vs_Lxy_overlay_{sample_name}.pdf"))

        # ---------------------------
        # Overlay efficiency vs pT
        # ---------------------------
        plt.clf()
        for label, jcol in jets_variants.items():
            matched_gen = jcol.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
            matched_gen = ak.drop_none(matched_gen)

            hist_pt_num = Hist(pt_axis)
            if ak.any(~ak.is_none(matched_gen)):
                hist_pt_num.fill(ak.flatten(matched_gen.pt, axis=None).compute())

            plot_efficiency(hist_pt_num, hist_pt_den, label=label)

        plt.ylim(top=1.05)
        plt.xlabel(r"$p_T$ (GeV) of GenVisStauTau")
        plt.title(f"Matching efficiency vs $p_T$ (all jet variants) — {sample_name}")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pT_output_dir, f"eff_vs_pt_overlay_{sample_name}.pdf"))
        '''

        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                            (cut_filtered_events.Jet.pt > 20)]
        jets_isTight_no_chHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                                            (cut_filtered_events.Jet.pt > 20) & \
                                                            (cut_filtered_events.Jet.isTight)]
        jets_isTight_chHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                                            (cut_filtered_events.Jet.pt > 20) & \
                                                            (cut_filtered_events.Jet.isTight) & \
                                                            (cut_filtered_events.Jet.chHEF > 0.01)]
        jets_isTightLV_no_chHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                                            (cut_filtered_events.Jet.pt > 20) & \
                                                            (cut_filtered_events.Jet.isTightLeptonVeto)]
        jets_isTightLV_chHEF = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                                            (cut_filtered_events.Jet.pt > 20) & \
                                                            (cut_filtered_events.Jet.isTightLeptonVeto) & \
                                                            (cut_filtered_events.Jet.chHEF > 0.01)]
    
        '''
        sorted_by_score = jets[ak.argsort(jets.disTauTag_score1, ascending=False)]
        jets = ak.singletons(ak.firsts(sorted_by_score))

        sorted_by_score = jets_isTight_no_chHEF[ak.argsort(jets_isTight_no_chHEF.disTauTag_score1, ascending=False)]
        jets_isTight_no_chHEF = ak.singletons(ak.firsts(sorted_by_score))

        sorted_by_score = jets_isTight_chHEF[ak.argsort(jets_isTight_chHEF.disTauTag_score1, ascending=False)]
        jets_isTight_chHEF = ak.singletons(ak.firsts(sorted_by_score))

        sorted_by_score = jets_isTightLV_no_chHEF[ak.argsort(jets_isTightLV_no_chHEF.disTauTag_score1, ascending=False)]
        jets_isTightLV_no_chHEF = ak.singletons(ak.firsts(sorted_by_score))

        sorted_by_score = jets_isTightLV_chHEF[ak.argsort(jets_isTightLV_chHEF.disTauTag_score1, ascending=False)]
        jets_isTightLV_chHEF = ak.singletons(ak.firsts(sorted_by_score))
        '''
        '''
        sorted_by_dxy_err = jets[ak.argsort(jets.dxy_err, ascending=True)]
        jets = ak.singletons(ak.firsts(sorted_by_dxy_err))

        sorted_by_dxy_err = jets_isTightLV_no_chHEF[ak.argsort(jets_isTightLV_no_chHEF.dxy_err, ascending=True)]
        jets_isTightLV_no_chHEF = ak.singletons(ak.firsts(sorted_by_dxy_err))

        sorted_by_dxy_err = jets_isTightLV_chHEF[ak.argsort(jets_isTightLV_chHEF.dxy_err, ascending=True)]
        jets_isTightLV_chHEF = ak.singletons(ak.firsts(sorted_by_dxy_err))
        '''
        jet_matched_gen_vis_taus = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus = ak.drop_none(jet_matched_gen_vis_taus)
        '''
        jet_matched_gen_vis_taus_isTight_no_chHEF = jets_isTight_no_chHEF.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_isTight_no_chHEF = ak.drop_none(jet_matched_gen_vis_taus_isTight_no_chHEF)

        jet_matched_gen_vis_taus_isTight_chHEF = jets_isTight_chHEF.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_isTight_chHEF = ak.drop_none(jet_matched_gen_vis_taus_isTight_chHEF)
        '''
        jet_matched_gen_vis_taus_isTightLV_no_chHEF = jets_isTightLV_no_chHEF.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_isTightLV_no_chHEF = ak.drop_none(jet_matched_gen_vis_taus_isTightLV_no_chHEF)

        jet_matched_gen_vis_taus_isTightLV_chHEF = jets_isTightLV_chHEF.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_isTightLV_chHEF = ak.drop_none(jet_matched_gen_vis_taus_isTightLV_chHEF)
    
        '''
        #############################################################################################################################################
        # eta efficiency hists  
        #############################################################################################################################################
        bins = np.arange(-2.1, 2.1, 0.1)
        eta_axis = axis.Variable(bins, name="eta")
        hist_eta_den = Hist(eta_axis)
        hist_eta_num = Hist(eta_axis)

        hist_eta_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.eta, axis=None).compute())
        hist_eta_num.fill(ak.flatten(jet_matched_gen_vis_taus.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num, hist_eta_den)

        #############################################################################################################################################
        # Lxy efficiency hists  
        #############################################################################################################################################
        bins = np.arange(0, 100, 5)
        Lxy_axis = axis.Variable(bins, name="Lxy")
        hist_Lxy_den = Hist(Lxy_axis)
        hist_Lxy_num = Hist(Lxy_axis)

        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy, axis=None).compute())
        hist_Lxy_num.fill(ak.flatten(jet_matched_gen_vis_taus.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num, hist_Lxy_den)
        '''

        #############################################################################################################################################
        # pT efficiency hists  
        #############################################################################################################################################
        pt_bins_low    = np.arange(20, 101, 20)
        pt_bins_med    = np.arange(100, 400, 30)
        pt_bins_high   = np.arange(400, 600, 40)
        pt_bins_higher = np.arange(600, 1000, 50)
        pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))

        pt_axis = axis.Variable(pt_bins_eff, flow=False, name="GenVisTau_pt")
        hist_pt_den = Hist(pt_axis)  # denominator: GenVisTau pT (selected)
        hist_pt_num = Hist(pt_axis)  # numerator: GenVisTau pT for those matched to isTightLV_no_chHEF jets

        # fill denominator with all selected GenVisStauTaus (your usual selection)
        hist_pt_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None).compute())

        # fill numerator with GenVisStauTaus that matched jets passing isTightLV_no_chHEF
        hist_pt_num.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_no_chHEF.pt, axis=None).compute())

        # sanitize & plot
        _sanitize_inplace(hist_pt_num, hist_pt_den)

        plt.clf()
        plot_efficiency(hist_pt_num, hist_pt_den)
        plt.title(f"isTightLV (no chHEF) efficiency vs GenVisTau pT — {sample_name}")
        plt.xlabel("GenVisTau pT [GeV]")
        plt.ylabel("Efficiency")
        plt.savefig(os.path.join(pT_output_dir, f"eff_vs_pt_iTLV_no_chHEF_{sample_name}.pdf"))

        '''
        var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="GenVisTau_pt")}
        hist_pt_den = Hist(var_axes['pt'])
        hist_pt_num = Hist(var_axes['pt'])

        hist_pt_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None).compute())
        hist_pt_num.fill(ak.flatten(jet_matched_gen_vis_taus.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num, hist_pt_den)
        '''

        '''
        #############################################################################################################################################
        # d0 efficiency hists  
        #############################################################################################################################################
        d0_bins = np.arange(0, 60, 5)
        d0_axis = axis.Variable(d0_bins, name="d0")
        hist_d0_den = Hist(d0_axis)
        hist_d0_num = Hist(d0_axis)

        hist_d0_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.d0, axis=None).compute())
        hist_d0_num.fill(ak.flatten(jet_matched_gen_vis_taus.d0, axis=None).compute())
        _sanitize_inplace(hist_d0_num, hist_d0_den)

        # ---- Append the "all_jets" payload to JSON (unchanged logic) ----
        _payload = {
            "eta": _hist_to_payload(hist_eta_num, hist_eta_den),
            "Lxy": _hist_to_payload(hist_Lxy_num, hist_Lxy_den),
            "pt":  _hist_to_payload(hist_pt_num,  hist_pt_den),
            "d0":  _hist_to_payload(hist_d0_num,  hist_d0_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "all_jets", _payload)
        '''

        '''
        # -------------------- isTight_no_chHEF --------------------
        hist_eta_num_isTight_no_chHEF = Hist(eta_axis)
        hist_eta_num_isTight_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_no_chHEF.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num_isTight_no_chHEF, hist_eta_den)

        hist_Lxy_num_isTight_no_chHEF = Hist(Lxy_axis)
        hist_Lxy_num_isTight_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_no_chHEF.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num_isTight_no_chHEF, hist_Lxy_den)

        hist_pt_num_isTight_no_chHEF = Hist(var_axes['pt'])
        hist_pt_num_isTight_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_no_chHEF.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num_isTight_no_chHEF, hist_pt_den)
        
        hist_d0_num_isTight_no_chHEF = Hist(d0_axis)
        hist_d0_num_isTight_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_no_chHEF.d0, axis=None).compute())
        _sanitize_inplace(hist_d0_num_isTight_no_chHEF, hist_d0_den)

        # JSON append
        _payload_isTight_no_chHEF = {
            #"eta": _hist_to_payload(hist_eta_num_isTight_no_chHEF,  hist_eta_den),
            #"Lxy": _hist_to_payload(hist_Lxy_num_isTight_no_chHEF,  hist_Lxy_den),
            #"pt":  _hist_to_payload(hist_pt_num_isTight_no_chHEF,   hist_pt_den),
            "d0":  _hist_to_payload(hist_d0_num_isTight_no_chHEF,   hist_d0_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "isTight_no_chHEF", _payload_isTight_no_chHEF)
        '''

        '''
        # -------------------- isTight_chHEF --------------------
        hist_eta_num_isTight_chHEF = Hist(eta_axis)
        hist_eta_num_isTight_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_chHEF.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num_isTight_chHEF, hist_eta_den)

        hist_Lxy_num_isTight_chHEF = Hist(Lxy_axis)
        hist_Lxy_num_isTight_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_chHEF.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num_isTight_chHEF, hist_Lxy_den)

        hist_pt_num_isTight_chHEF = Hist(var_axes['pt'])
        hist_pt_num_isTight_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_chHEF.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num_isTight_chHEF, hist_pt_den)
        
        hist_d0_num_isTight_chHEF = Hist(d0_axis)
        hist_d0_num_isTight_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTight_chHEF.d0, axis=None).compute())
        _sanitize_inplace(hist_d0_num_isTight_chHEF, hist_d0_den)

        _payload_isTight_chHEF = {
            #"eta": _hist_to_payload(hist_eta_num_isTight_chHEF,  hist_eta_den),
            #"Lxy": _hist_to_payload(hist_Lxy_num_isTight_chHEF,  hist_Lxy_den),
            #"pt":  _hist_to_payload(hist_pt_num_isTight_chHEF,   hist_pt_den),
            "d0":  _hist_to_payload(hist_d0_num_isTight_chHEF,   hist_d0_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "isTight_chHEF", _payload_isTight_chHEF)
        '''
        '''
        # -------------------- isTightLV_no_chHEF --------------------
        hist_eta_num_isTightLV_no_chHEF = Hist(eta_axis)
        hist_eta_num_isTightLV_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_no_chHEF.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num_isTightLV_no_chHEF, hist_eta_den)

        hist_Lxy_num_isTightLV_no_chHEF = Hist(Lxy_axis)
        hist_Lxy_num_isTightLV_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_no_chHEF.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num_isTightLV_no_chHEF, hist_Lxy_den)

        hist_pt_num_isTightLV_no_chHEF = Hist(var_axes['pt'])
        hist_pt_num_isTightLV_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_no_chHEF.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num_isTightLV_no_chHEF, hist_pt_den)

        hist_d0_num_isTightLV_no_chHEF = Hist(d0_axis)
        hist_d0_num_isTightLV_no_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_no_chHEF.d0, axis=None).compute())
        _sanitize_inplace(hist_d0_num_isTightLV_no_chHEF, hist_d0_den)

        _payload_isTightLV_no_chHEF = {
            "eta": _hist_to_payload(hist_eta_num_isTightLV_no_chHEF,  hist_eta_den),
            "Lxy": _hist_to_payload(hist_Lxy_num_isTightLV_no_chHEF,  hist_Lxy_den),
            "pt":  _hist_to_payload(hist_pt_num_isTightLV_no_chHEF,   hist_pt_den),
            "d0":  _hist_to_payload(hist_d0_num_isTightLV_no_chHEF,   hist_d0_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "isTightLV_no_chHEF", _payload_isTightLV_no_chHEF)
        
        # -------------------- isTightLV_chHEF --------------------
        hist_eta_num_isTightLV_chHEF = Hist(eta_axis)
        hist_eta_num_isTightLV_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_chHEF.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num_isTightLV_chHEF, hist_eta_den)

        hist_Lxy_num_isTightLV_chHEF = Hist(Lxy_axis)
        hist_Lxy_num_isTightLV_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_chHEF.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num_isTightLV_chHEF, hist_Lxy_den)

        hist_pt_num_isTightLV_chHEF = Hist(var_axes['pt'])
        hist_pt_num_isTightLV_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_chHEF.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num_isTightLV_chHEF, hist_pt_den)

        hist_d0_num_isTightLV_chHEF = Hist(d0_axis)
        hist_d0_num_isTightLV_chHEF.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLV_chHEF.d0, axis=None).compute())
        _sanitize_inplace(hist_d0_num_isTightLV_chHEF, hist_d0_den)

        _payload_isTightLV_chHEF = {
            "eta": _hist_to_payload(hist_eta_num_isTightLV_chHEF,  hist_eta_den),
            "Lxy": _hist_to_payload(hist_Lxy_num_isTightLV_chHEF,  hist_Lxy_den),
            "pt":  _hist_to_payload(hist_pt_num_isTightLV_chHEF,   hist_pt_den),
            "d0":  _hist_to_payload(hist_d0_num_isTightLV_chHEF,   hist_d0_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "isTightLV_chHEF", _payload_isTightLV_chHEF)
        '''

        '''
        # --- JETS (category="jets") ---
        # eta
        bins = np.arange(-2.1, 2.1, 0.1)
        eta_axis = axis.Variable(bins, name="eta")
        hist_eta_den = Hist(eta_axis); hist_eta_num = Hist(eta_axis)
        hist_eta_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.eta, axis=None).compute())
        hist_eta_num.fill(ak.flatten(jet_matched_gen_vis_taus.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num, hist_eta_den)

        # Lxy
        bins = np.arange(0, 100, 5)
        Lxy_axis = axis.Variable(bins, name="Lxy")
        hist_Lxy_den = Hist(Lxy_axis); hist_Lxy_num = Hist(Lxy_axis)
        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy, axis=None).compute())
        hist_Lxy_num.fill(ak.flatten(jet_matched_gen_vis_taus.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num, hist_Lxy_den)

        # pT
        pt_bins_low = np.arange(20, 101, 20)
        pt_bins_med = np.arange(100, 400, 30)
        pt_bins_high = np.arange(400, 600, 40)
        pt_bins_higher = np.arange(600, 1000, 50)
        pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))
        var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="GenVisTau_pt")}
        hist_pt_den = Hist(var_axes['pt']); hist_pt_num = Hist(var_axes['pt'])
        hist_pt_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None).compute())
        hist_pt_num.fill(ak.flatten(jet_matched_gen_vis_taus.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num, hist_pt_den)

        # JSON append for 'jets'
        _payload = {
            "eta": _hist_to_payload(hist_eta_num, hist_eta_den),
            "Lxy": _hist_to_payload(hist_Lxy_num, hist_Lxy_den),
            "pt":  _hist_to_payload(hist_pt_num,  hist_pt_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "all_jets", _payload)
        
        # --- isTight (category="isTight") ---
        # eta
        bins = np.arange(-2.1, 2.1, 0.1)
        eta_axis = axis.Variable(bins, name="eta")
        hist_eta_den = Hist(eta_axis); hist_eta_num = Hist(eta_axis)
        hist_eta_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.eta, axis=None).compute())
        hist_eta_num.fill(ak.flatten(jet_matched_gen_vis_taus_isTight.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num, hist_eta_den)

        # Lxy
        bins = np.arange(0, 100, 5)
        Lxy_axis = axis.Variable(bins, name="Lxy")
        hist_Lxy_den = Hist(Lxy_axis); hist_Lxy_num = Hist(Lxy_axis)
        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy, axis=None).compute())
        hist_Lxy_num.fill(ak.flatten(jet_matched_gen_vis_taus_isTight.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num, hist_Lxy_den)

        # pT
        var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="GenVisTau_pt")}
        hist_pt_den = Hist(var_axes['pt']); hist_pt_num = Hist(var_axes['pt'])
        hist_pt_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None).compute())
        hist_pt_num.fill(ak.flatten(jet_matched_gen_vis_taus_isTight.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num, hist_pt_den)

        # JSON append for 'isTight'
        _payload = {
            "eta": _hist_to_payload(hist_eta_num, hist_eta_den),
            "Lxy": _hist_to_payload(hist_Lxy_num, hist_Lxy_den),
            "pt":  _hist_to_payload(hist_pt_num,  hist_pt_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "isTight", _payload)

        # --- isTightLeptonVeto (category="isTightLeptonVeto") ---
        # eta
        bins = np.arange(-2.1, 2.1, 0.1)
        eta_axis = axis.Variable(bins, name="eta")
        hist_eta_den = Hist(eta_axis); hist_eta_num = Hist(eta_axis)
        hist_eta_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.eta, axis=None).compute())
        hist_eta_num.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLeptonVeto.eta, axis=None).compute())
        _sanitize_inplace(hist_eta_num, hist_eta_den)

        # Lxy
        bins = np.arange(0, 100, 5)
        Lxy_axis = axis.Variable(bins, name="Lxy")
        hist_Lxy_den = Hist(Lxy_axis); hist_Lxy_num = Hist(Lxy_axis)
        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy, axis=None).compute())
        hist_Lxy_num.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLeptonVeto.parent.Lxy, axis=None).compute())
        _sanitize_inplace(hist_Lxy_num, hist_Lxy_den)

        # pT
        var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="GenVisTau_pt")}
        hist_pt_den = Hist(var_axes['pt']); hist_pt_num = Hist(var_axes['pt'])
        hist_pt_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None).compute())
        hist_pt_num.fill(ak.flatten(jet_matched_gen_vis_taus_isTightLeptonVeto.pt, axis=None).compute())
        _sanitize_inplace(hist_pt_num, hist_pt_den)

        # JSON append for 'isTightLeptonVeto'
        _payload = {
            "eta": _hist_to_payload(hist_eta_num, hist_eta_den),
            "Lxy": _hist_to_payload(hist_Lxy_num, hist_Lxy_den),
            "pt":  _hist_to_payload(hist_pt_num,  hist_pt_den),
        }
        _append_histograms_to_json(json_cache, sample_name, "isTightLeptonVeto", _payload)
        '''
