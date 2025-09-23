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

#filenames = {}
#for i in range(67):
    #filenames[f"Stau_100_10mm_{i}"] = f"root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_{i}_0.root"

# Load the file
filenames = {
    #'Stau_100_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_200_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_200_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_200_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_200_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_1mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
}

PFNanoAODSchema.mixins["DisMuon"] = "Muon"
samples = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()
    #eventsnotselected_samples[sample_name] = NanoEventsFactory.from_root(
        #{files: "EventsNotSelected"},
        #schemaclass=NanoAODSchema,
    #).events()

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array: 
    mval = first.metric_table(second) 
    return ak.all(mval > threshold, axis=-1)

#deltaR_dict = {}
out_dir_genmuon_high_score     = "deltaR_GenMuon_plots"
os.makedirs(out_dir_genmuon_high_score, exist_ok=True)

os.makedirs("jets_not_matched_isTight", exist_ok=True)
os.makedirs("jets_matched_isTight", exist_ok=True)

os.makedirs("jets_not_matched_isTight_has_GenVisStauTau", exist_ok=True)
os.makedirs("jets_not_matched_isTight_has_no_GenVisStauTau", exist_ok=True)

os.makedirs("jets_2nd_highest_score_matched_GenVisStauTau", exist_ok=True)

output_dir_dR = "dR_between_jets"
os.makedirs(output_dir_dR, exist_ok=True)

output_dir_partonFlavour = "partonFlavour_jets"
os.makedirs(output_dir_partonFlavour, exist_ok=True)

def _to_np_flat(arr):
    # Works for awkward and dask-awkward
    if hasattr(arr, "compute"):
        arr = arr.compute()
    return ak.to_numpy(ak.flatten(arr, axis=None))

'''
def _overlay_two_1d(a1, a2, bins, rng, xlabel, title, outpath, l1="highest (not matched)", l2="second (matched)"):
    x1 = _to_np_flat(a1)
    x2 = _to_np_flat(a2)
    plt.figure()
    plt.hist(x1, bins=bins, range=rng, histtype="step", lw=2, label=l1)
    plt.hist(x2, bins=bins, range=rng, histtype="step", lw=2, label=l2)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
'''
def _overlay_two_1d(a1, a2, bins, rng, xlabel, title, outpath,
                    l1="highest (not matched)", l2="second (matched)",
                    return_counts=False):
    x1 = _to_np_flat(a1)
    x2 = _to_np_flat(a2)

    plt.figure()
    n1, be1, _ = plt.hist(x1, bins=bins, range=rng, histtype="step", lw=2, label=l1)
    n2, be2, _ = plt.hist(x2, bins=bins, range=rng, histtype="step", lw=2, label=l2)
    plt.xlabel(xlabel); plt.ylabel("Counts"); plt.title(title)
    plt.grid(True, ls="--", alpha=0.5); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

    if return_counts:
        # return counts and the shared bin edges
        return n1, n2, be1

def _hist2d_pair(xarr, yarr, bins, rng, xlabel, title, outpath, log=True):
    # flatten + (dask-)awkward -> numpy
    if hasattr(xarr, "compute"): xarr = xarr.compute()
    if hasattr(yarr, "compute"): yarr = yarr.compute()
    x = ak.to_numpy(ak.flatten(xarr, axis=None))
    y = ak.to_numpy(ak.flatten(yarr, axis=None))

    # keep finite
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0 or y.size == 0:
        print(f"[warn] empty for {title}, skipping.")
        return

    plt.figure()
    plt.hist2d(
        x, y,
        bins=[bins, bins],            # same binning for x & y
        range=[rng, rng],             # same ranges for x & y
        norm=mcolors.LogNorm() if log else None,
    )
    # diagonal reference
    plt.plot([rng[0], rng[1]], [rng[0], rng[1]], ls="--", lw=1, color="k")
    plt.xlabel(f"{xlabel} (highest not matched)")
    plt.ylabel(f"{xlabel} (second matched)")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Counts")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ---------- what to plot (field, bins, (min,max), label) ----------
plots = [
    #("pt",                 60, (0, 750),     r"Jet $p_T$ [GeV]"),
    #("eta",                60, (-2.5, 2.5),  r"Jet $\eta$"),
    #("phi",                64, (-3.2, 3.2),  r"Jet $\phi$"),
    #("mass",               60, (0, 120),     "Jet mass [GeV]"),
    #("area",               50, (0, 1.5),     "Jet area"),
    #("disTauTag_score1",   50, (0, 1.0),     "disTauTag_score1"),
    #("disTauTag_score0",   50, (0, 1.0),     "disTauTag_score0"),
    #("btagPNetTauVJet",    50, (0, 1.0),     "btagPNetTauVJet"),
    #("btagDeepFlavQG",     50, (0, 1.0),     "btagDeepFlavQG"),
    #("btagPNetQvG",        50, (0, 1.0),     "btagPNetQvG"),
    #("muEF",               50, (0, 0.8),     "muEF"),
    #("chHEF",              50, (0, 1.0),     "chHEF"),
    #("neHEF",              50, (0, 1.0),     "neHEF"),
    #("chEmEF",             50, (0, 1.0),     "chEmEF"),
    #("neEmEF",             50, (0, 1.0),     "neEmEF"),
    #("nConstituents",      80, (0, 80),      "nConstituents"),
    #("chMultiplicity",     60, (0, 60),      "chMultiplicity"),
    #("neMultiplicity",     60, (0, 60),      "neMultiplicity"),
    #("qgl",                50, (0, 1.0),     "qgl"),
    #("puIdDisc",           60, (-1, 1),      "puIdDisc"),
    #("puId",                8, (-0.5, 7.5),  "puId"),
    #("jetId",               8, (-0.5, 7.5),  "jetId"),
    ("dxy",                80, (0, 30),     "dxy"),
]

# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        # add dxy to jet fields
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = abs(ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, \
                ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)))
        events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")
        vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.parent.vx
        vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.parent.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        parent_with_Lxy = ak.with_field(events.GenVisTau.parent, Lxy, where="Lxy")
        events['GenVisTau'] = ak.with_field(events.GenVisTau, parent_with_Lxy, where="parent")
        
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

        #events = events[(ak.num(events.GenVisStauTaus) > 0)]

        events['GenMuon'] = gpart[(abs(gpart.pdgId) == 13) & (gpart.hasFlags("isLastCopy"))] 
        events['GenMuon'] = events.GenMuon[(events.GenMuon.pt > 20) & (abs(events.GenMuon.eta) < 2.4)]

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

        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                            (cut_filtered_events.Jet.pt > 20) & \
                                            (cut_filtered_events.Jet.isTightLeptonVeto)]

        '''
        new_var_jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                            (cut_filtered_events.Jet.pt > 20)]
        vals = new_var_jets.dxy
        if hasattr(vals, "compute"):  # dask-awkward safe
            vals = vals.compute()
        arr = ak.to_numpy(ak.flatten(vals, axis=None))

        plt.figure()
        plt.hist(arr, bins=3, range=(998, 1001), histtype='step', lw=2)
        plt.xlabel(r"Jet $|d_{xy}|$")   # adjust units if known
        plt.ylabel("Counts")
        plt.title(f"{sample_name} — all jets |dxy|")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("plots", f"{sample_name}_jets_all_dxy.pdf"))
        plt.close()
        '''

        has_2_jets = ak.num(jets) == 2
        jets_2j = jets[has_2_jets]
        cut_filtered_events_2j = cut_filtered_events[has_2_jets]

        sorted_by_score_2j = jets_2j[ak.argsort(jets_2j.disTauTag_score1, ascending=False)]
        highest_score_jets = ak.singletons(sorted_by_score_2j[:, 0])
        second_highest_score_jets = ak.singletons(sorted_by_score_2j[:, 1])

        highest_not_matched_mask = delta_r_mask(highest_score_jets, cut_filtered_events_2j.GenVisStauTaus, 0.4) 
        second_not_matched_mask  = delta_r_mask(second_highest_score_jets, cut_filtered_events_2j.GenVisStauTaus, 0.4)

        evt_keep = ak.flatten(highest_not_matched_mask & (~second_not_matched_mask), axis=1)

        highest_not_matched = ak.firsts(highest_score_jets[evt_keep])              
        second_matched      = ak.firsts(second_highest_score_jets[evt_keep])

        '''
        highest_not_matched = highest_not_matched[highest_not_matched.muEF > 0.5]
        genmuon_kept = cut_filtered_events_2j.GenMuon[evt_keep]
        mu_mask = (highest_not_matched.muEF > 0.5)

        jets_hi_mu = highest_not_matched[mu_mask]   
        muons_mu   = genmuon_kept[mu_mask]

        dR_GenMuon = jets_hi_mu.metric_table(muons_mu)

        arr = dR_GenMuon.compute() if hasattr(dR_GenMuon, "compute") else dR_GenMuon
        arr = ak.fill_none(arr, np.nan)
        x = ak.to_numpy(ak.firsts(ak.firsts(arr)))

        plt.figure()
        plt.hist(x, bins=60, range=(0, 5.0), histtype="step", lw=2)
        plt.xlabel(r"$\Delta R(\mathrm{jet}, \mu)$")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: ΔR(jet, GenMuon)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_out, f"{sample_name}_dR_jet_GenMuon.pdf"))
        plt.close()
        '''
        
        sample_out = os.path.join("compare_highestNotMatched_vs_secondMatched", sample_name)
        os.makedirs(sample_out, exist_ok=True)

        '''
        for field, nb, rng, xlabel in plots:
            if hasattr(highest_not_matched, field) and hasattr(second_matched, field):
                _overlay_two_1d(
                    getattr(highest_not_matched, field),
                    getattr(second_matched, field),
                    bins=nb,
                    rng=rng,
                    xlabel=xlabel,
                    title=f"{sample_name}: highest(not matched) vs second(matched) — {field}",
                    outpath=os.path.join(sample_out, f"{sample_name}_{field}.pdf"),
                )
        '''

        for field, nb, rng, xlabel in plots:
            if hasattr(highest_not_matched, field) and hasattr(second_matched, field):
                outpath = os.path.join(sample_out, f"{sample_name}_{field}.pdf")
                want = (field == "dxy")  # only compute integrals for dxy (or set True for all)

                ret = _overlay_two_1d(
                    getattr(highest_not_matched, field),
                    getattr(second_matched, field),
                    bins=nb, rng=rng, xlabel=xlabel,
                    title=f"{sample_name}: highest(not matched) vs second(matched) — {field}",
                    outpath=outpath,
                    return_counts=want,
                )

                if want:
                    n1, n2, edges = ret
                    full_highest = int(np.sum(n1))
                    full_second  = int(np.sum(n2))
                    excl1_highest = int(np.sum(n1[1:]))  
                    excl1_second  = int(np.sum(n2[1:]))

                    first_bin_range = f"[{edges[0]:.3g}, {edges[1]:.3g})"
                    rest_range      = f"[{edges[1]:.3g}, {edges[-1]:.3g})"

                    print(f"[{sample_name}] {field} histogram (range {rng})")
                    print(f"  Binning: {len(edges)-1} bins; first bin = {first_bin_range}")

                    print("  --- All bins included ---")
                    print(f"    highest not matched: {full_highest}")
                    print(f"    second matched     : {full_second}")
                    print(f"    difference         : {full_second - full_highest}")

                    print("  --- Excluding the FIRST bin ---")
                    print(f"    highest not matched, bins {rest_range}: {excl1_highest}")
                    print(f"    second matched,      bins {rest_range}: {excl1_second}")
                    print(f"    difference, excluding first bin       : {excl1_second - excl1_highest}")   
        '''
        sample_out2d = os.path.join("compare_highestNotMatched_vs_secondMatched_2D", sample_name)
        os.makedirs(sample_out2d, exist_ok=True)

        # make the 2D histograms
        for field, nb, rng, xlabel in plots:
            if hasattr(highest_not_matched, field) and hasattr(second_matched, field):
                _hist2d_pair(
                    getattr(highest_not_matched, field),
                    getattr(second_matched, field),
                    bins=nb,
                    rng=rng,
                    xlabel=xlabel,
                    title=f"{sample_name}: 2D — {field}",
                    outpath=os.path.join(sample_out2d, f"{sample_name}_{field}_2D.pdf"),
                    log=True,
                )
        '''

        #jets = jets[jets.disTauTag_score1 > 0.90]

        #jets_all = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.disTauTag_score1 > 0.9)]

        # add isTight to jets if lepton veto needed
        #jets_tight = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTight) & (cut_filtered_events.Jet.disTauTag_score1 > 0.9)]
        
        # add isTightLeptonVeto to jets if lepton veto needed
        #jets_tightLeptonVeto = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTightLeptonVeto) & (cut_filtered_events.Jet.disTauTag_score1 > 0.9)]
        '''
        ###################################################################################################
        # Plots for deltaR for GenMuon wrt jets
        ###################################################################################################
        deltaR_all = jets_all.metric_table(cut_filtered_events.GenMuon).compute()
        deltaR_tight = jets_tight.metric_table(cut_filtered_events.GenMuon).compute()
        deltaR_tightLeptonVeto = jets_tightLeptonVeto.metric_table(cut_filtered_events.GenMuon).compute()
        
        plt.figure()
        bins = np.linspace(0, 5, 49)

        plt.hist(ak.ravel(deltaR_all), bins=bins, histtype='step', lw=2, label='All jets')
        plt.hist(ak.ravel(deltaR_tight), bins=bins, histtype='step', lw=2, label='isTight')
        plt.hist(ak.ravel(deltaR_tightLeptonVeto), bins=bins, histtype='step', lw=2, label='isTightLeptonVeto')

        plt.xlabel(r'$\Delta R$(jet, GenMuon)')
        plt.ylabel("Number of jet-muon pairs")
        plt.title(r'$\Delta R$ between jets w/score > 0.9 and GenMuons')
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_genmuon_high_score, f"deltaR_GenMuon_{sample_name}.pdf"))
        plt.close()
        '''

        '''
        # Sort the selected jets by disTauTag_score1 (descending) and take the first jet per event
        sorted_by_score = jets[ak.argsort(jets.disTauTag_score1, ascending=False)]
        highest_score_jets = ak.singletons(ak.firsts(sorted_by_score))

        jets_matched = cut_filtered_events.GenVisStauTaus.nearest(highest_score_jets, threshold=0.4)
        jets_not_matched = highest_score_jets[delta_r_mask(highest_score_jets, cut_filtered_events.GenVisStauTaus,   0.4)]

        # Get the second highest scoring jet per event
        second_highest_score_jets = ak.singletons(sorted_by_score_2j[:, 1])
        highest_score_jets        = ak.singletons(sorted_by_score_2j[:, 0])

        # Match GenVisStauTaus to second highest scoring jets
        jets_matched_second_highest_score = cut_filtered_events_2j.GenVisStauTaus.nearest(second_highest_score_jets, threshold=0.4)

        # Select second-highest jets that were NOT matched
        #jets_not_matched_second_highest_score = second_highest_score_jets[delta_r_mask(second_highest_score_jets, cut_filtered_events.GenVisStauTaus, 0.4)]

        is_matched_to_second = ak.num(jets_matched_second_highest_score) > 0
        #score_matched_2nd_jet = ak.flatten(second_highest_score_jets[is_matched_to_second].disTauTag_score1.compute())
        #score_top_jet_in_matched_to_2nd = ak.flatten(highest_score_jets[is_matched_to_second].disTauTag_score1.compute())

        score_matched_2nd_jet = second_highest_score_jets[is_matched_to_second]
        score_top_jet_in_matched_to_2nd = highest_score_jets[is_matched_to_second]

        bins = np.arange(0, 25, 1)
        plt.hist(ak.flatten(score_top_jet_in_matched_to_2nd.partonFlavour).compute(), bins=bins, histtype='step', lw=2, label='Highest Score (not matched)', color='tab:blue')
        plt.hist(ak.flatten(score_matched_2nd_jet.partonFlavour).compute(), bins=bins, histtype='step', lw=2, label='2nd Score (matched)', color='tab:orange')

        plt.xlabel("partonFlavour pdgId")
        plt.ylabel("Counts")
        plt.title("Jet Score Parton Flavour Comparison")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir_partonFlavour}/partonFlavour_jets_{sample_name}.pdf")
        plt.close()
        '''
        
        '''
        dR_between_jets = score_matched_2nd_jet.metric_table(score_top_jet_in_matched_to_2nd)

        bins = np.arange(0, 4, 0.1)

        plt.hist(ak.flatten(dR_between_jets).compute(), bins=bins, histtype='step', lw=2)

        plt.xlabel("dR")
        plt.ylabel("Counts")
        plt.title("dR between highest score and 2nd highest")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir_dR}/dR_between_jets_{sample_name}.pdf")
        plt.close()
        '''

        '''
        #has_gen_vis_stau_tau = ak.num(cut_filtered_events.GenVisStauTaus) > 0
        has_no_gen_vis_stau_tau = ak.num(cut_filtered_events.GenVisStauTaus) == 0
        has_unmatched_jet = ak.num(jets_not_matched) > 0
        #selected_event_mask = has_gen_vis_stau_tau & has_unmatched_jet
        selected_event_mask_no_GenVisTau = has_no_gen_vis_stau_tau & has_unmatched_jet
        #jets_not_matched = jets_not_matched[selected_event_mask]
        jets_not_matched = jets_not_matched[selected_event_mask_no_GenVisTau]

        score_diff = abs(score_matched_2nd_jet - score_top_jet_in_matched_to_2nd)

        plt.figure()
        bins = np.linspace(0, 1, 50)

        plt.hist(score_matched_2nd_jet, bins=bins, histtype='step', lw=2, label='2nd Score (matched)', color='tab:blue')
        plt.hist(score_top_jet_in_matched_to_2nd, bins=bins, histtype='step', lw=2, label='Top Score (no match)', color='tab:orange')

        # Score difference between matched 2nd and top jet
        plt.hist(score_diff, bins=bins, histtype='step', lw=2, label='|2nd - Top Score|', color='tab:green')

        plt.xlabel("disTauTag Score")
        plt.ylabel("Counts")
        plt.title("Jet Score Comparison")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"jets_2nd_highest_score_matched_GenVisStauTau/{sample_name}_ScoreComparison_AllCurves.pdf")
        plt.close()
        '''


        '''
        plt.hist(score_diff, bins=bins, histtype='step', lw=2, label='|2nd - Top Score|')

        plt.xlabel("Absolute disTauTag Score Difference")
        plt.ylabel("Counts")
        plt.title("Score Difference: 2nd Matched vs Highest Score Jet")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{sample_name}_ScoreComparison_MatchedToSecondHighestDiff.pdf")
        plt.close()
        '''


        '''
        plt.figure()
        plt.hist(ak.to_numpy(ak.flatten(jets_not_matched.pt.compute())), bins=60, range=(0, 750), histtype='step', lw=2)
        plt.xlabel(r"Not Matched Jet $p_T$ [GeV]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet $p_T$")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("jets_not_matched_isTight_has_GenVisStauTau", f"{sample_name}_JetPt_NotMatched.pdf"))
        plt.close()

        plt.figure()
        plt.hist(ak.to_numpy(ak.flatten(jets_not_matched.eta.compute())), bins=60, range=(-3, 3), histtype='step', lw=2)
        plt.xlabel(r"Not Matched Jet $\eta$")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet $\eta$")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("jets_not_matched_isTight_has_GenVisStauTau", f"{sample_name}_JetEta_NotMatched.pdf"))
        plt.close()
        
        plt.figure()
        plt.hist(ak.to_numpy(ak.flatten(jets_not_matched.dxy.compute())), bins=60, range=(0, 10), histtype='step', lw=2)
        plt.xlabel(r"Not Matched Jet $d_{xy}$ [cm]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet dxy")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("jets_not_matched_isTight_has_GenVisStauTau", f"{sample_name}_JetDxy_NotMatched.pdf"))
        plt.close()
        
        plt.figure()
        plt.hist(ak.to_numpy(ak.flatten(jets_not_matched.disTauTag_score1.compute())), bins=60, range=(0, 1), histtype='step', lw=2)
        plt.xlabel(r"Not Matched Jet disTauTag Score 1")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet disTauTag Score 1")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("jets_not_matched_isTight_has_GenVisStauTau", f"{sample_name}_JetScore_NotMatched.pdf"))
        plt.close()
        
        plt.figure()
        plt.hist(
            ak.to_numpy(ak.flatten(jets_not_matched.dxy.compute())),
            bins=np.arange(0, 1.05, 0.05),
            histtype='step',
            lw=2
        )
        plt.xlabel(r"Jet $d_{xy}$ [cm]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet $d_{{xy}}$ (Zoomed)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("jets_not_matched_isTight_has_GenVisStauTau", f"{sample_name}_JetDxyZoom_NotMatched.pdf"))
        plt.close()

        plt.figure()
        plt.hist(ak.to_numpy(ak.flatten(jets_not_matched.disTauTag_score1.compute())), bins=60, range=(0, 1), histtype='step', lw=2)
        plt.xlabel(r"Not Matched Jet disTauTag Score 1")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} Jet disTauTag Score 1")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("jets_not_matched_isTight_has_no_GenVisStauTau", f"{sample_name}_JetScore_NotMatched.pdf"))
        plt.close()
        '''