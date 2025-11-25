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
    #'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v10/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
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
def _overlay_two_1d(
        a1, a2, bins, rng, xlabel, title, outpath,
        l1="highest (not matched)", l2="second (matched)",
        ylog=False
    ):
        x1 = _to_np_flat(a1)
        x2 = _to_np_flat(a2)
        plt.figure()
        plt.hist(x1, bins=bins, range=rng, histtype="step", lw=2, label=l1)
        plt.hist(x2, bins=bins, range=rng, histtype="step", lw=2, label=l2)
        plt.xlabel(xlabel)
        plt.ylabel("Counts")
        plt.title(title)
        if ylog:   # <-- new option
            plt.yscale("log")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
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
'''
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

def _hist2d_xy(xarr, yarr, xbins, ybins, xrange, yrange,
               xlabel, ylabel, title, outpath, log=True):
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
        bins=[xbins, ybins],
        range=[xrange, yrange],
        norm=mcolors.LogNorm() if log else None,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Counts")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ---------- what to plot (field, bins, (min,max), label) ----------
plots = [
    ("pt",                 60, (0, 750),     r"Jet $p_T$ [GeV]"),
    ("eta",                60, (-2.5, 2.5),  r"Jet $\eta$"),
    ("phi",                64, (-3.2, 3.2),  r"Jet $\phi$"),
    ("mass",               60, (0, 120),     "Jet mass [GeV]"),
    ("area",               50, (0, 1.5),     "Jet area"),
    ("disTauTag_score1",   50, (0, 1.0),     "disTauTag_score1"),
    ("disTauTag_score0",   50, (0, 1.0),     "disTauTag_score0"),
    ("btagPNetTauVJet",    50, (0, 1.0),     "btagPNetTauVJet"),
    ("btagDeepFlavQG",     50, (0, 1.0),     "btagDeepFlavQG"),
    ("btagPNetQvG",        50, (0, 1.0),     "btagPNetQvG"),
    ("muEF",               50, (0, 0.8),     "muEF"),
    ("chHEF",              50, (0, 1.0),     "chHEF"),
    ("neHEF",              50, (0, 1.0),     "neHEF"),
    ("chEmEF",             50, (0, 1.0),     "chEmEF"),
    ("neEmEF",             50, (0, 1.0),     "neEmEF"),
    ("nConstituents",      80, (0, 80),      "nConstituents"),
    ("chMultiplicity",     60, (0, 60),      "chMultiplicity"),
    ("neMultiplicity",     60, (0, 60),      "neMultiplicity"),
    ("qgl",                50, (0, 1.0),     "qgl"),
    ("puIdDisc",           60, (-1, 1),      "puIdDisc"),
    ("puId",                8, (-0.5, 7.5),  "puId"),
    #("jetId",               8, (-0.5, 7.5),  "jetId"),
    ("dxy",                100, (0, 20),     "dxy [cm]"),
]

# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        #print(len(events.compute()))
        # add dxy to jet fields
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = abs(ak.where(ak.all(events.Jet.constituents.pf.charge == 0, axis = -1), -999, \
                ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)))
        events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")
        dxy_err = abs(ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0Err, axis = 2))
        events['Jet'] = ak.with_field(events.Jet, dxy_err, where="dxy_err")

        # build 4-vector for jets and store energy
        Jet_vec = ak.zip(
            {
                "pt":  events.Jet.pt,
                "eta": events.Jet.eta,
                "phi": events.Jet.phi,
                "mass": events.Jet.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=coffea.nanoevents.methods.vector.behavior,
        )
        events["Jet"] = ak.with_field(events.Jet, Jet_vec.energy, where="energy")

        pf = events.Jet.constituents.pf

        pf_vec = ak.zip(
            {
                "pt":  pf.pt,
                "eta": pf.eta,
                "phi": pf.phi,
                "mass": pf.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=coffea.nanoevents.methods.vector.behavior,
        )
        pf_with_p = ak.with_field(pf, pf_vec.p, where="p")
        consts = events.Jet.constituents
        consts = ak.with_field(consts, pf_with_p, where="pf")
        events["Jet"] = ak.with_field(events.Jet, consts, where="constituents")

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
        events['GenMuon'] = events.GenMuon[(events.GenMuon.pt > 20) & \
                                            (abs(events.GenMuon.eta) < 2.4) & \
                                            (events.GenMuon.distinctParent.distinctParent.pdgId == 1000015)]

        vx = events.GenMuon.vx
        vy = events.GenMuon.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        events['GenMuon'] = ak.with_field(events.GenMuon, Lxy, where="Lxy")

        GenMuon_d0 = abs((events.GenMuon.vy - events.GenVtx.y) * np.cos(events.GenMuon.phi) - \
              (events.GenMuon.vx - events.GenVtx.x) * np.sin(events.GenMuon.phi))
        events['GenMuon'] = ak.with_field(events.GenMuon, GenMuon_d0, where="d0")

        events['GenElectron'] = events.GenPart[(abs(events.GenPart.pdgId) == 11) & (events.GenPart.hasFlags("isLastCopy"))] 

        vx = events.GenElectron.vx
        vy = events.GenElectron.vy
        electron_Lxy = np.sqrt(vx**2 + vy**2)
        events['GenElectron'] = ak.with_field(events.GenElectron, electron_Lxy, where="Lxy")
        GenElectron_d0 = abs((events.GenElectron.vy - events.GenVtx.y) * np.cos(events.GenElectron.phi) - \
              (events.GenElectron.vx - events.GenVtx.x) * np.sin(events.GenElectron.phi))
        events['GenElectron'] = ak.with_field(events.GenElectron, GenElectron_d0, where="d0")

        events['GenElectron'] = events.GenElectron[(events.GenElectron.pt > 20) & \
                                                    (abs(events.GenElectron.eta) < 2.4) & \
                                                    (events.GenElectron.Lxy < 100.0) & \
                                                    (events.GenElectron.distinctParent.distinctParent.pdgId == 1000015)]

        
        mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenMuon) == 1) & (ak.num(events.GenElectron) == 0)
        events = events[mask]
        '''
        mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenElectron) == 1) & (ak.num(events.GenMuon) == 0)
        events = events[mask]
        '''

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
                                            (cut_filtered_events.Jet.neHEF < 0.99) & \
                                            (cut_filtered_events.Jet.neEmEF < 0.9) & \
                                            ((cut_filtered_events.Jet.chMultiplicity + cut_filtered_events.Jet.neMultiplicity) > 1) & \
                                            (cut_filtered_events.Jet.chMultiplicity > 0) & \
                                            (cut_filtered_events.Jet.muEF < 0.8) & \
                                            (cut_filtered_events.Jet.chEmEF < 0.8)]

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

        gen_sel = cut_filtered_events_2j.GenVisStauTaus[evt_keep]
        gen_electron = cut_filtered_events_2j.GenElectron[evt_keep]
        gen_muon = cut_filtered_events_2j.GenMuon[evt_keep]
        cut_filtered_events_2j = cut_filtered_events_2j[evt_keep]
        #total_events_before = int(ak.num(cut_filtered_events_2j, axis=0).compute())

        '''
        # leading PF cand (charged) for highest-score jet NOT matched
        pf_high = highest_not_matched.constituents.pf
        charged_pf_high = pf_high[pf_high.charge != 0]
        sorted_by_p_high = charged_pf_high[ak.argsort(charged_pf_high.p, ascending=False)]
        lead_pf_high = ak.firsts(sorted_by_p_high)     # first (highest-p) PF per jet
        
        # leading PF cand (charged) for second-highest-score jet that IS matched
        pf_second = second_matched.constituents.pf
        charged_pf_second = pf_second[pf_second.charge != 0]
        sorted_by_p_second = charged_pf_second[ak.argsort(charged_pf_second.p, ascending=False)]
        lead_pf_second = ak.firsts(sorted_by_p_second)
        
        # PF candidates for each jet category
        pf_high   = lead_pf_high
        pf_second = lead_pf_second

        valid_high   = ~ak.is_none(pf_high.hcalFraction)
        valid_second = ~ak.is_none(pf_second.hcalFraction)

        hcal_mask = ((pf_high.hcalFraction   > 0.2) & (pf_second.hcalFraction > 0.2))

        # apply the mask to the leading PF candidates
        pf_high_masked   = pf_high[hcal_mask]
        pf_second_masked = pf_second[hcal_mask]
        '''

        '''
        # --- caloFraction distributions (leading PF cand level, masked) ---

        calofrac_high   = pf_high_masked.caloFraction
        calofrac_second = pf_second_masked.caloFraction

        _overlay_two_1d(
            calofrac_high,
            calofrac_second,
            bins=50,
            rng=(0.0, 1.0),
            xlabel="PF candidate caloFraction (lead PF, hcalFraction>0.2 on both jets)",
            title="Leading PF caloFraction: highest (not matched) vs second (matched)\n(hcalFraction>0.2 for both leading PFs)",
            outpath=f"caloFraction_highest_vs_second_hcalgt02_{sample_name}.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )

        # --- hcalFraction distributions (leading PF cand level, masked) ---

        hcalfrac_high   = pf_high_masked.hcalFraction
        hcalfrac_second = pf_second_masked.hcalFraction

        _overlay_two_1d(
            hcalfrac_high,
            hcalfrac_second,
            bins=50,
            rng=(0.0, 1.0),
            xlabel="PF candidate hcalFraction (lead PF, hcalFraction>0.2 on both jets)",
            title="Leading PF hcalFraction: highest (not matched) vs second (matched)\n(hcalFraction>0.2 for both leading PFs)",
            outpath=f"hcalFraction_highest_vs_second_hcalgt02_{sample_name}.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )
        '''

        '''
        lead_pf_high_clean   = pf_high[~ak.is_none(pf_high)]
        lead_pf_second_clean = pf_second[~ak.is_none(pf_second)]

        pdgid_high_awk   = lead_pf_high_clean.pdgId.compute()
        pdgid_second_awk = lead_pf_second_clean.pdgId.compute()

        pdgid_high_np   = ak.to_numpy(pdgid_high_awk)
        pdgid_second_np = ak.to_numpy(pdgid_second_awk)

        # absolute value if you want |PDGID|
        abs_pdgid_high   = np.abs(pdgid_high_np)
        abs_pdgid_second = np.abs(pdgid_second_np)

        mask_high   = abs_pdgid_high   <= 211
        mask_second = abs_pdgid_second <= 211

        abs_pdgid_high_plot   = abs_pdgid_high[mask_high]
        abs_pdgid_second_plot = abs_pdgid_second[mask_second]
        
        # integer-aligned bins from 1–211 (bin edges at 0.5, 1.5, ..., 210.5, 211.5)
        abs_pdg_max_plot = 211

        _overlay_two_1d(
            abs_pdgid_high_plot,
            abs_pdgid_second_plot,
            bins=abs_pdg_max_plot,          # 211 bins
            rng=(0.5, abs_pdg_max_plot + 0.5),
            xlabel="|PDG ID| of leading charged PF candidate",
            title="Leading charged PF |PDG ID|: highest (not matched) vs second (matched)",
            outpath=f"leadPF_absPdgId_highest_vs_second_{sample_name}.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )
        
        _overlay_two_1d(
            abs_pdgid_high_plot,
            abs_pdgid_second_plot,
            bins=40,
            rng=(0.5, 40.5),
            xlabel="|PDG ID| of leading charged PF candidate",
            title="Leading charged PF |PDG ID| (zoomed lepton region)",
            outpath=f"leadPF_absPdgId_highest_vs_second_zoom_{sample_name}.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )
        '''

        '''
        # --- caloFraction distributions (PF-candidate level) ---

        calofrac_high   = pf_high.caloFraction
        calofrac_second = pf_second.caloFraction

        _overlay_two_1d(
            calofrac_high,
            calofrac_second,
            bins=50,
            rng=(0.0, 1.0),
            xlabel="PF candidate caloFraction",
            title="PF caloFraction: highest (not matched) vs second (matched)",
            outpath=f"caloFraction_highest_vs_second_{sample_name}.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )

        # --- hcalFraction distributions (PF-candidate level) ---

        hcalfrac_high   = pf_high.hcalFraction
        hcalfrac_second = pf_second.hcalFraction

        _overlay_two_1d(
            hcalfrac_high,
            hcalfrac_second,
            bins=50,
            rng=(0.0, 1.0),
            xlabel="PF candidate hcalFraction",
            title="PF hcalFraction: highest (not matched) vs second (matched)",
            outpath=f"hcalFraction_highest_vs_second_{sample_name}.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )
        '''

        '''
        n_pf_high   = ak.num(highest_not_matched.constituents.pf, axis=-1)
        n_pf_second = ak.num(second_matched.constituents.pf, axis=-1)

        _overlay_two_1d(
            n_pf_high,
            n_pf_second,
            bins=80,
            rng=(0, 80),
            xlabel="Number of PF candidates in jet",
            title="PF candidate multiplicity: highest (not matched) vs second (matched)",
            outpath=f"nPFcands_highest_vs_second_{sample_name}.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )
        '''

        '''
        # plot dR between GenElectron and jets for highest not matched vs 2nd matched
        dR_GenElectron = highest_not_matched.metric_table(gen_electron).compute()
        dr_ele_flat = ak.to_numpy(ak.ravel(dR_GenElectron))

        dR_GenElectron_second = second_matched.metric_table(gen_electron).compute()
        dr_ele_flat_second = ak.to_numpy(ak.ravel(dR_GenElectron_second))

        out_dir = os.path.join("deltaR_GenElectron", sample_name)
        os.makedirs(out_dir, exist_ok=True)

        bins = np.linspace(0.0, 5.0, 51)

        plt.figure()
        plotted = False

        if dr_ele_flat.size:
            plt.hist(
                dr_ele_flat,
                bins=bins,
                histtype="step",
                lw=2,
                label="highest_not_matched vs GenElectron",
            )
            plotted = True

        if dr_ele_flat_second.size:
            plt.hist(
                dr_ele_flat_second,
                bins=bins,
                histtype="step",
                lw=2,
                label="second_matched vs GenElectron",
            )
            plotted = True

        plt.xlabel(r"$\Delta R$(jet, GenElectron)")
        plt.ylabel("Number of jet–electron pairs")
        plt.title(f"{sample_name}: $\Delta R$ between jets and GenElectrons")
        if plotted:
            plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_name}_deltaR_GenElectron.pdf"))
        plt.close()
        '''

        '''
        # leading PF cand (charged) for highest-score jet NOT matched
        pf_high = highest_not_matched.constituents.pf
        charged_pf_high = pf_high[pf_high.charge != 0]
        sorted_by_p_high = charged_pf_high[ak.argsort(charged_pf_high.p, ascending=False)]
        lead_pf_high = ak.firsts(sorted_by_p_high)     # first (highest-p) PF per jet
        lead_p_high = lead_pf_high.p                   # |p| of leading PF cand

        # leading PF cand (charged) for second-highest-score jet that IS matched
        pf_second = second_matched.constituents.pf
        charged_pf_second = pf_second[pf_second.charge != 0]
        sorted_by_p_second = charged_pf_second[ak.argsort(charged_pf_second.p, ascending=False)]
        lead_pf_second = ak.firsts(sorted_by_p_second)
        lead_p_second = lead_pf_second.p

        # ratios: jet energy / leading-PF momentum
        ratio_high = lead_p_high / highest_not_matched.energy
        ratio_second = lead_p_second / second_matched.energy        

        _overlay_two_1d(
            ratio_high,
            ratio_second,
            bins=50,
            rng=(0, 5),
            xlabel=r"$p_{\mathrm{lead\ PF}} / E_{\mathrm{jet}}$",
            title=r"Leading PF momentum / jet energy: highest (not matched) vs second (matched)",
            outpath="p_over_E_leadPF_highest_vs_second_varBins.pdf",
            l1="highest (not matched)",
            l2="second (matched)",
        )
        '''
        '''
        hcalfrac_high   = lead_pf_high.hcalFraction
        hcalfrac_second = lead_pf_second.hcalFraction

        _hist2d_xy(
            ratio_high,
            hcalfrac_high,
            xbins=50,
            ybins=50,
            xrange=(0.0, 5.0),
            yrange=(0.0, 1.0),
            xlabel=r"$p_{\mathrm{lead\ PF}} / E_{\mathrm{jet}}$",
            ylabel="PF candidate hcalFraction",
            title=r"Highest (not matched): $p_{\mathrm{lead\ PF}} / E_{\mathrm{jet}}$ vs hcalFraction",
            outpath=f"pOverE_vs_hcal_highest_{sample_name}.pdf",
        )

        # 2D: p/E vs hcalFraction for second (matched)
        _hist2d_xy(
            ratio_second,
            hcalfrac_second,
            xbins=50,
            ybins=50,
            xrange=(0.0, 5.0),
            yrange=(0.0, 1.0),
            xlabel=r"$p_{\mathrm{lead\ PF}} / E_{\mathrm{jet}}$",
            ylabel="PF candidate hcalFraction",
            title=r"Second (matched): $p_{\mathrm{lead\ PF}} / E_{\mathrm{jet}}$ vs hcalFraction",
            outpath=f"pOverE_vs_hcal_second_{sample_name}.pdf",
        )
        '''
        '''
        pf_ele_mask = (abs(highest_not_matched.constituents.pf.pdgId) == 11)
        new_mask = (ak.sum(pf_ele_mask, axis=-1) > 0)
        cut_filtered_events_2j = cut_filtered_events_2j[new_mask]
        ele_sel = (cut_filtered_events_2j.Electron.isPFcand)

        # Per-event counts of True/False
        n_true_per_event  = ak.sum(ele_sel,  axis=-1)              # number of True per event
        n_false_per_event = ak.sum(~ele_sel, axis=-1)              # number of False per event

        # Compute to concrete arrays for counting/printing
        n_true_per_event_np  = ak.to_numpy(n_true_per_event.compute())
        n_false_per_event_np = ak.to_numpy(n_false_per_event.compute())

        # Totals across events
        n_events          = int(n_true_per_event_np.shape[0])
        events_with_true  = int((n_true_per_event_np  > 0).sum())
        events_with_false = int((n_false_per_event_np > 0).sum())

        print(f"Total events: {total_events_before}")
        print(f"Total events with pf cands pdgID==11: {n_events}")
        print(f"Events containing at least one TRUE:  {events_with_true}")
        print(f"Events containing at least one FALSE: {events_with_false}")
        '''
        '''
        pf_by_jet = highest_not_matched.constituents.pf   # shape: [event, jet, *pf*]

        # Explicitly take jet0 and jet1, then concatenate along the PF-cand axis
        pf0 = pf_by_jet[:, 0]
        pf1 = pf_by_jet[:, 1]
        # (defensive) replace Nones with empty lists so concatenate never chokes
        pf0 = ak.fill_none(pf0, [], axis=0)
        pf1 = ak.fill_none(pf1, [], axis=0)
        pf_2jets = ak.concatenate([pf0, pf1], axis=1)     # shape: [event, *pf* from both jets]

        # Gen electrons
        gen = cut_filtered_events_2j.GenElectron

        # Build 4-vectors (massless is fine for ΔR)
        gen4 = ak.zip({"pt":gen.pt, "eta":gen.eta, "phi":gen.phi, "mass":ak.zeros_like(gen.pt)},
                      with_name="Momentum4D")
        pf4  = ak.zip({"pt":pf_2jets.pt, "eta":pf_2jets.eta, "phi":pf_2jets.phi, "mass":ak.zeros_like(pf_2jets.pt)},
                      with_name="Momentum4D")

        # All gen–pf pairs per event; compute ΔR and choose nearest PF per GenElectron
        pairs = ak.cartesian({"gen_i": gen4, "pf_j": pf4, "pf_obj": pf_2jets}, axis=1)  # axis=1 = per event cross-product
        dR = pairs["gen_i"].deltaR(pairs["pf_j"])
        best_idx = ak.argmin(dR, axis=-1)

        # For each GenElectron, the PF cand it's "constructed as"
        best_pf = pairs["pf_obj"][ak.local_index(dR, axis=-1), best_idx]  # equivalent to "take the pf with min ΔR per GenElectron"
        best_pdgid = best_pf.pdgId
        best_dR = ak.min(dR, axis=-1)

        # Quick check: how often is it a charged pion?
        is_charged_pion = (ak.abs(best_pdgid) == 211)
        frac_pion = ak.mean(is_charged_pion)

        print("Fraction of GenElectrons whose nearest PF cand is a charged pion (|pdgId|==211):", float(frac_pion))

        # Optional: counts by PF pdgId
        pdg_flat = ak.to_numpy(ak.flatten(best_pdgid, axis=None))
        vals, counts = np.unique(pdg_flat, return_counts=True)
        print("Counts by PF pdgId:", dict(zip(vals, counts)))
        '''

        '''
        pf_mu_mask = (abs(highest_not_matched.constituents.pf.pdgId) == 13)
        new_mask = (ak.sum(pf_mu_mask, axis=-1) > 0)
        cut_filtered_events_2j = cut_filtered_events_2j[new_mask]
        mu_sel = (cut_filtered_events_2j.Muon.isGlobal & cut_filtered_events_2j.Muon.isPFcand)

        # Per-event counts of True/False
        n_true_per_event  = ak.sum(mu_sel,  axis=-1)              
        n_false_per_event = ak.sum(~mu_sel, axis=-1)              

        # Compute to concrete arrays for counting/printing
        n_true_per_event_np  = ak.to_numpy(n_true_per_event.compute())
        n_false_per_event_np = ak.to_numpy(n_false_per_event.compute())

        # Totals across events
        n_events          = int(n_true_per_event_np.shape[0])
        events_with_true  = int((n_true_per_event_np  > 0).sum())
        events_with_false = int((n_false_per_event_np > 0).sum())

        print(f"Total events: {total_events_before}")
        print(f"Total events with pf cands pdgID==13: {n_events}")
        print(f"Events containing at least one TRUE:  {events_with_true}")
        print(f"Events containing at least one FALSE: {events_with_false}")

        no_mu_mask = ~new_mask 
        pf_pdg_no_mu = highest_not_matched.constituents.pf.pdgId[no_mu_mask]
        gen_mu_no_pf = gen_muon[no_mu_mask]
        pf_pdg_no_mu_list = ak.to_list(pf_pdg_no_mu.compute())
        pts_list  = ak.to_list(gen_mu_no_pf.pt.compute())
        d0s_list  = ak.to_list(gen_mu_no_pf.d0.compute())
        lxys_list = ak.to_list(gen_mu_no_pf.Lxy.compute())
        def _fmt(v):
            return "None" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.2f}"

        print(f"Events with NO PF muon (|pdgId|==13): {len(pf_pdg_no_mu_list)}")
        for i, (ids, pts, d0s, lxys) in enumerate(zip(pf_pdg_no_mu_list, pts_list, d0s_list, lxys_list)):
            triples = [f"(pt={_fmt(p)}, d0={_fmt(d0)}, Lxy={_fmt(lxy)})"
                       for p, d0, lxy in zip(pts, d0s, lxys)]
            print(f"EventNoPFMuon[{i}] pdgIds: {ids} | GenMuon: {', '.join(triples) if triples else '[]'}")
        '''

        
        # plot dR between GenMuon and jets for highest not matched vs 2nd matched 
        highest_not_matched = highest_not_matched[highest_not_matched.muEF < 0.8]
        mu_mask = (highest_not_matched.muEF < 0.8)

        jets_hi_mu_EF = highest_not_matched[mu_mask]
        muons_hi_muEF = gen_muon[mu_mask]

        second_mu_mask = (second_matched.muEF < 0.8)
        jets_second_hi_mu_EF = second_matched[second_mu_mask]
        muons_second_hi_muEF = gen_muon[second_mu_mask]

        dR_GenMuon = jets_hi_mu_EF.metric_table(muons_hi_muEF).compute()
        dr_mu_flat = ak.to_numpy(ak.ravel(dR_GenMuon))

        dR_GenMuon_second = jets_second_hi_mu_EF.metric_table(muons_second_hi_muEF).compute()
        dr_mu_flat_second  = ak.to_numpy(ak.ravel(dR_GenMuon_second)) 

        out_dir = os.path.join("deltaR_GenMuon", sample_name)
        os.makedirs(out_dir, exist_ok=True)

        bins = np.linspace(0.0, 5.0, 51)

        plt.figure()
        plotted = False

        if dr_mu_flat.size:
            plt.hist(dr_mu_flat, bins=bins, histtype="step", lw=2,
                     label="highest_not_matched vs GenMuon")
            plotted = True

        if dr_mu_flat_second.size:
            plt.hist(dr_mu_flat_second, bins=bins, histtype="step", lw=2,
                     label="second_matched vs GenMuon")
            plotted = True
        plt.xlabel(r"$\Delta R$(jet, GenMuon)")
        plt.ylabel("Number of jet–muon pairs")
        plt.title(f"{sample_name}: $\Delta R$ between jets and GenMuons if (μEF<0.8)")
        if plotted:
            plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_name}_deltaR_GenMuon_v13.pdf"))
        plt.close()
        

        '''
        # --- Leading PF candidate selection for highest_not_matched jets ---
        sorted_pf_high = highest_not_matched.constituents.pf[
            ak.argsort(highest_not_matched.constituents.pf.pt, ascending=False)
        ]
        highest_pf_cand_high = ak.firsts(sorted_pf_high, axis=1)

        # --- Leading PF candidate selection for second_matched jets ---
        sorted_pf_second = second_matched.constituents.pf[
            ak.argsort(second_matched.constituents.pf.pt, ascending=False)
        ]
        highest_pf_cand_second = ak.firsts(sorted_pf_second, axis=1)

        # --- Now do ΔR metric_table between each jet and its own leading PF cand ---
        dR_highest = highest_not_matched.metric_table(highest_pf_cand_high, axis=None).compute()
        dR_second  = second_matched.metric_table(highest_pf_cand_second, axis=None).compute()

        # Flatten for plotting
        dR_highest_flat = ak.to_numpy(ak.ravel(dR_highest))
        dR_second_flat  = ak.to_numpy(ak.ravel(dR_second))

        sample_out = os.path.join("compare_highestNotMatched_vs_secondMatched", sample_name)
        os.makedirs(sample_out, exist_ok=True)

        # --- Plot 1D overlay ---
        bins_1d = np.linspace(0.0, 0.8, 81)
        plt.figure()
        if dR_highest_flat.size:
            plt.hist(dR_highest_flat, bins=bins_1d, histtype='step', lw=2, label='highest_not_matched (leading PF)')
        if dR_second_flat.size:
            plt.hist(dR_second_flat,  bins=bins_1d, histtype='step', lw=2, label='second_matched (leading PF)')
        plt.xlabel(r'$\Delta R$(jet, leading PF cand)')
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: ΔR to leading PF candidate")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_out, f"{sample_name}_deltaR_leadingPF_1D_overlay.pdf"))
        plt.close()
        '''

        '''
        sample_out = os.path.join("compare_highestNotMatched_vs_secondMatched", sample_name)
        os.makedirs(sample_out, exist_ok=True)

        dR_highest = highest_not_matched.metric_table(highest_not_matched.constituents.pf).compute()
        dR_second  = second_matched.metric_table(second_matched.constituents.pf).compute()

        dR_highest_flat = ak.to_numpy(ak.ravel(dR_highest))
        dR_second_flat  = ak.to_numpy(ak.ravel(dR_second))

        bins_1d = np.linspace(0.0, 0.8, 81)  # adjust to your jet R if needed

        plt.figure()
        if dR_highest_flat.size:
            plt.hist(dR_highest_flat, bins=bins_1d, histtype='step', lw=2, label='highest_not_matched')
        if dR_second_flat.size:
            plt.hist(dR_second_flat,  bins=bins_1d, histtype='step', lw=2, label='second_matched')
        plt.xlabel(r'$\Delta R$(jet, PF constituent)')
        plt.ylabel("Number of jet–PF pairs")
        plt.title(f"{sample_name}: ΔR between jets and PF constituents")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_out, f"{sample_name}_deltaR_jet_pfconstituents_1D_overlay.pdf"))
        plt.close() 
        '''    

        '''
        sample_out = os.path.join("compare_highestNotMatched_vs_secondMatched", sample_name)
        os.makedirs(sample_out, exist_ok=True)

        # --- Helper to flatten → numpy (handles jagged + dask)
        def _flat_np(x):
            return ak.to_numpy(ak.flatten(x, axis=None).compute())

        # --- PF constituent pT overlay (highest_not_matched vs second_matched)
        pt_highest = _flat_np(highest_not_matched.constituents.pf.pt)
        pt_second  = _flat_np(second_matched.constituents.pf.pt)

        if pt_highest.size + pt_second.size > 0:
            all_pts = np.concatenate([pt_highest, pt_second]) if pt_highest.size and pt_second.size else (pt_highest if pt_highest.size else pt_second)
            # pick a sane upper edge (cap extreme tails)
            max_pt = float(np.percentile(all_pts, 99.5)) if all_pts.size else 50.0
            max_pt = max(50.0, max_pt)
            bins_pt = np.linspace(0.0, max_pt, 60)

            plt.figure()
            if pt_highest.size:
                plt.hist(pt_highest, bins=bins_pt, histtype="step", lw=2, label="highest_not_matched")
            if pt_second.size:
                plt.hist(pt_second,  bins=bins_pt, histtype="step", lw=2, label="second_matched")
            plt.xlabel("PF constituent $p_T$ [GeV]")
            plt.ylabel("Counts")
            plt.title(f"{sample_name}: PF-constituent $p_T$ (highest_not_matched vs second_matched)")
            plt.legend()
            plt.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(sample_out, f"{sample_name}_pfconst_pt_overlay.pdf"))
            plt.close()

        # --- Charged-only: numberOfPixelHits overlay
        charged_highest_mask = (highest_not_matched.constituents.pf.charge != 0)
        charged_second_mask  = (second_matched.constituents.pf.charge != 0)

        pix_highest = _flat_np(highest_not_matched.constituents.pf.numberOfPixelHits[charged_highest_mask])
        pix_second  = _flat_np(second_matched.constituents.pf.numberOfPixelHits[charged_second_mask])

        if pix_highest.size + pix_second.size > 0:
            max_hits = int(max(pix_highest.max() if pix_highest.size else 0,
                               pix_second.max()  if pix_second.size  else 0))
            bins_hits = np.arange(-0.5, max_hits + 0.5 + 1, 1)

            plt.figure()
            if pix_highest.size:
                plt.hist(pix_highest, bins=bins_hits, histtype="step", lw=2, label="highest_not_matched (charged)")
            if pix_second.size:
                plt.hist(pix_second,  bins=bins_hits, histtype="step", lw=2, label="second_matched (charged)")
            plt.xlabel("PF constituent numberOfPixelHits (charged only)")
            plt.ylabel("Counts")
            plt.title(f"{sample_name}: numberOfPixelHits (highest_not_matched vs second_matched)")
            plt.legend()
            plt.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(sample_out, f"{sample_name}_pfconst_pixelHits_charged_overlay.pdf"))
            plt.close()
        '''


        '''
        taus_keep = cut_filtered_events_2j.staus_taus[evt_keep]

        children = taus_keep.distinctChildren 
        ele_mask = (abs(children.pdgId) == 11)

        n_ele_per_tau = ak.sum(ele_mask, axis=-1)
        n_ele_per_event = ak.sum(n_ele_per_tau, axis=-1)

        if hasattr(n_ele_per_event, "compute"):
            n_ele_per_event = n_ele_per_event.compute()
        n_ele_per_event = np.asarray(n_ele_per_event)

        # ---- Plot: number of electrons from tau decay per event ----
        sample_out = os.path.join("compare_highestNotMatched_vs_secondMatched", sample_name)
        os.makedirs(sample_out, exist_ok=True)

        edges = np.arange(-0.5, 3.5 + 1e-9, 1)  # bins centered at 0,1,2,3 (adjust if needed)

        plt.figure()
        plt.hist(n_ele_per_event, bins=edges, histtype='step', lw=2)
        plt.xlabel("Electrons from tau decay per event")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: e from τ decay (events: highest NOT matched & second matched)")
        plt.xticks(np.arange(0, 4, 1))
        plt.xlim(-0.5, 3.5)
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_out, f"{sample_name}_nElectrons_fromTauDecay.pdf"))
        plt.close()

        mu_mask = (abs(children.pdgId) == 13)
        n_mu_per_tau    = ak.sum(mu_mask, axis=-1)
        n_mu_per_event  = ak.sum(n_mu_per_tau, axis=-1)
        if hasattr(n_mu_per_event, "compute"):
            n_mu_per_event = n_mu_per_event.compute()
        n_mu_per_event = np.asarray(n_mu_per_event)

        edges = np.arange(-0.5, 3.5 + 1e-9, 1)
        plt.figure()
        plt.hist(n_mu_per_event, bins=edges, histtype='step', lw=2)
        plt.xlabel("Muons from tau decay per event")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: μ from τ decay (events: highest NOT matched & second matched)")
        plt.xticks(np.arange(0, 4, 1))
        plt.xlim(-0.5, 3.5)
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_out, f"{sample_name}_nMuons_fromTauDecay.pdf"))
        plt.close()
        '''

        '''
        highest_matched_mask = ~highest_not_matched_mask

        evt_keep_highest_strict = ak.flatten(highest_matched_mask & second_not_matched_mask, axis=1)

        highest_matched        = ak.firsts(highest_score_jets[evt_keep_highest_strict])
        second_not_matched     = ak.firsts(second_highest_score_jets[evt_keep_highest_strict])
        genvistau_sel_highest  = cut_filtered_events_2j.GenVisStauTaus[evt_keep_highest_strict]
        genmu_sel_highest      = cut_filtered_events_2j.GenMuon[evt_keep_highest_strict]
        '''

        '''
        # Make output dir (same style as your other plots)
        sample_out = os.path.join("compare_highestNotMatched_vs_secondMatched", sample_name)
        os.makedirs(sample_out, exist_ok=True)

        # Plot |pdgId| with integer-centered bins 0..500
        plt.figure()
        plt.hist(to_plot, bins=np.arange(-0.5, 500.5 + 1, 1), histtype="step", lw=2)
        plt.xlabel(r"|pdgId| of GenVisStauTau")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: GenVisStauTaus |pdgId| (0–500)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_out, f"{sample_name}_GenVisStauTau_absPdgId_0to500.pdf"))
        plt.close()
        '''
        
        '''
        sel_events = cut_filtered_events_2j[evt_keep]
        def _to_np_1d(arr):
            if hasattr(arr, "compute"):
                arr = arr.compute()
            return np.asarray(arr)

        def _to_list_per_event(arr):
            if hasattr(arr, "compute"):
                arr = arr.compute()
            return ak.to_list(arr)

        runs   = _to_np_1d(sel_events.run)
        lumis  = _to_np_1d(sel_events.luminosityBlock)
        evids  = _to_np_1d(sel_events.event)

        # per-event list of both jet pTs (the two jets passing your cuts)
        jets_pts_all = _to_list_per_event(jets_2j[evt_keep].pt)
        jets_eta_all = _to_list_per_event(jets_2j[evt_keep].eta)

        pt_highest_not_matched = _to_np_1d(highest_not_matched.pt)
        pt_second_matched      = _to_np_1d(second_matched.pt)

        eta_highest_not_matched = _to_np_1d(highest_not_matched.eta)
        eta_second_matched      = _to_np_1d(second_matched.eta)

        print(f"[{sample_name}] Events where HIGHEST jet is NOT matched and SECOND jet IS matched: {len(evids)}")
        for i in range(len(evids)):
            r, ls, ev = runs[i], lumis[i], evids[i]
            all_pts = jets_pts_all[i]  # list of two pTs for this event
            print(
                f"{i+1:5d}: run={r}  lumiSection={ls}  eventId={ev} | "
                f"highest_not_matched_pt={pt_highest_not_matched[i]:.1f} | "
                f"second_matched_pt={pt_second_matched[i]:.1f}"
                f"highest_not_matched_eta={eta_highest_not_matched[i]:.1f} | "
                f"second_matched_eta={eta_second_matched[i]:.1f}"
            )
        '''

        '''
        out_dir = os.path.join("GenElectron_all", sample_name)
        os.makedirs(out_dir, exist_ok=True)

        # Flatten to 1D
        d0_all  = ak.to_numpy(ak.flatten(gen_electron.d0,  axis=None).compute())
        Lxy_all = ak.to_numpy(ak.flatten(gen_electron.Lxy, axis=None).compute())

        # d0: 0–20
        bins_d0 = np.linspace(0.0, 20.0, 41)
        plt.figure()
        if d0_all.size:
            plt.hist(d0_all, bins=bins_d0, histtype="step", lw=2, label="All GenElectrons")
        plt.xlabel(r"GenElectron $d_0$ [cm]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: GenElectron $d_0$ (all)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_name}_GenElectron_d0_all.pdf"))
        plt.close()

        # Lxy: 0–40
        bins_Lxy = np.linspace(0.0, 40.0, 41)
        plt.figure()
        if Lxy_all.size:
            plt.hist(Lxy_all, bins=bins_Lxy, histtype="step", lw=2, label="All GenElectrons")
        plt.xlabel(r"GenElectron $L_{xy}$ [cm]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: GenElectron $L_{{xy}}$ (all)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_name}_GenElectron_Lxy_all.pdf"))
        plt.close()
        '''

        '''
        out_dir = os.path.join("GenMuon_all", sample_name)
        os.makedirs(out_dir, exist_ok=True)

        # Flatten to 1D
        d0_all  = ak.to_numpy(ak.flatten(gen_muon.d0,  axis=None).compute())
        Lxy_all = ak.to_numpy(ak.flatten(gen_muon.Lxy, axis=None).compute())

        # d0: 0–20
        bins_d0 = np.linspace(0.0, 20.0, 41)
        plt.figure()
        if d0_all.size:
            plt.hist(d0_all, bins=bins_d0, histtype="step", lw=2, label="All GenMuons")
        plt.xlabel(r"GenMuon $d_0$ [cm]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: GenMuon $d_0$ (all)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_name}_GenMuon_d0_all.pdf"))
        plt.close()

        # Lxy: 0–40
        bins_Lxy = np.linspace(0.0, 40.0, 41)
        plt.figure()
        if Lxy_all.size:
            plt.hist(Lxy_all, bins=bins_Lxy, histtype="step", lw=2, label="All GenMuons")
        plt.xlabel(r"GenMuon $L_{xy}$ [cm]")
        plt.ylabel("Counts")
        plt.title(f"{sample_name}: GenMuon $L_{{xy}}$ (all)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sample_name}_GenMuon_Lxy_all.pdf"))
        plt.close()
        '''

        '''
        sel_events = cut_filtered_events_2j[evt_keep]

        def _to_np_1d(arr):
            if hasattr(arr, "compute"):
                arr = arr.compute()
            return np.asarray(arr)

        runs  = _to_np_1d(sel_events.run)
        lumis = _to_np_1d(sel_events.luminosityBlock)
        evids = _to_np_1d(sel_events.event)

        # Print to screen
        print(f"[{sample_name}] Selected events (highest NOT matched, second matched): {len(evids)}")
        for i, (r, ls, ev) in enumerate(zip(runs, lumis, evids), start=1):
            print(f"{i:5d}: run={r}  lumiSection={ls}  eventId={ev}")
        '''

        '''
        vals = second_matched.matched_gen.partonFlavour
        if hasattr(vals, "compute"):  # dask-awkward safe
            vals = vals.compute()
        arr = ak.to_numpy(ak.flatten(vals, axis=None))

        plt.figure()
        edges = np.arange(0, 25, 1)  # edges: 0,1,...,24
        plt.hist(arr, bins=edges, histtype='step', lw=2)
        plt.xlabel("partonFlavour")
        plt.ylabel("Counts")
        plt.title(f"{sample_name} — second_matched.matched_gen.partonFlavour")
        plt.xticks(np.arange(0, 25, 1))
        plt.xlim(0, 24)
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_out, f"{sample_name}_second_matched.matched_gen.partonFlavour.pdf"))
        plt.close()
        '''
        '''
        sample_out = os.path.join("compare_highestNotMatched_vs_secondMatched", sample_name)
        os.makedirs(sample_out, exist_ok=True)

        for field, nb, rng, xlabel in plots:
            if hasattr(highest_not_matched, field) and hasattr(second_matched, field):
                _overlay_two_1d(
                    getattr(highest_not_matched, field),
                    getattr(second_matched, field),
                    bins=nb,
                    rng=rng,
                    xlabel=xlabel,
                    title=f"{sample_name}: highest(not matched) vs second(matched) — {field}",
                    outpath=os.path.join(sample_out, f"{sample_name}_{field}_require_GenMuon.pdf"),
                )
        '''

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
        '''

        '''       
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
        plt.savefig(os.path.join(out_dir_genmuon_high_score, f"deltaR_GenMuon_{sample_name}_requireGenMuon.pdf"))
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