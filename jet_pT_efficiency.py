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
    """Get the ratio (efficiency) of the passing over passing + failing probes.
    NaN values are replaced with 0.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
         : hist.Hist
            The histogram of the denominator.

    Returns
    -------
        ratio : hist.Hist
            The ratio histogram.
        yerr : numpy.ndarray
            The y error of the ratio histogram.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = passing_probes.values(flow=True) / denominator.values(flow=True)
                                 
    ratio = hist.Hist(hist.Hist(*passing_probes.axes))
    ratio[:] = np.nan_to_num(ratio_values)
    yerr = intervals.ratio_uncertainty(passing_probes.values(), denominator.values(), uncertainty_type="efficiency")

    return ratio, yerr


def plot_efficiency(passing_probes, denominator, log=False, **kwargs):
    """Plot the efficiency using the ratio of passing to passing + failing probes.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        denominator : hist.Hist
            The histogram of the denominator
        **kwargs
            Keyword arguments to pass to hist.Hist.plot1d.

    Returns
    -------
        List[Hist1DArtists]

    """
    ratio_hist, yerr = get_ratio_histogram(passing_probes, denominator)
    plt.ylabel('efficiency')
    if log:  plt.xscale('log')
    return ratio_hist.plot1d(histtype="errorbar", yerr=yerr, xerr=True, flow="none", **kwargs)

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
    #'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_1mm_no_cut'    : 'root://cmseos.fnal.gov//store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_10mm_no_cut'   : 'root://cmseos.fnal.gov//store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_100mm_no_cut'  : 'root://cmseos.fnal.gov//store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1mm_no_cut'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_10mm_no_cut'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm_no_cut'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1mm_no_cut'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_10mm_no_cut'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_100mm_no_cut'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
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

os.makedirs(pT_output_dir, exist_ok=True)
os.makedirs(Lxy_output_dir, exist_ok=True)
os.makedirs(eta_output_dir, exist_ok=True)

if __name__ == '__main__':
    for sample_name, events in samples.items():
        print(f"Processing sample: {sample_name}")
        # add dxy to jet fields
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)
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
                                                        (abs(events.GenVisTau.eta) < 2.1)]

        events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
        staus_taus = events['staus_taus']

        mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        mask_tauh = ~mask_taul

        one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
        one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

        filtered_events = events[one_tauh_evt & one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau
    
        tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus_taus.eta) < 2.1), axis=-1)
        num_taus = ak.num(filtered_events.staus_taus[tau_selections])
        num_tau_mask = num_taus > 1
        cut_filtered_events = filtered_events[num_tau_mask]

        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.1) & (cut_filtered_events.Jet.pt > 20)]
        jets_isTight = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.1) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTight) & (cut_filtered_events.Jet.chHEF > 0.01)]
        jets_isTightLeptonVeto = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.1) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTightLeptonVeto) & (cut_filtered_events.Jet.chHEF > 0.01)]

        jet_matched_gen_vis_taus = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus = ak.drop_none(jet_matched_gen_vis_taus)

        jet_matched_gen_vis_taus_isTight = jets_isTight.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_isTight = ak.drop_none(jet_matched_gen_vis_taus_isTight)

        jet_matched_gen_vis_taus_isTightLeptonVeto = jets_isTightLeptonVeto.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        jet_matched_gen_vis_taus_isTightLeptonVeto = ak.drop_none(jet_matched_gen_vis_taus_isTightLeptonVeto)

        #############################################################################################################################################
        # eta efficency hists
        #############################################################################################################################################
        bins = np.arange(-2.1, 2.1, 0.1)

        eta_axis = axis.Variable(bins, name="eta")
        hist_eta_den = Hist(eta_axis)
        hist_eta_num = Hist(eta_axis)
            
        hist_eta_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.eta, axis=None).compute())
        hist_eta_num.fill(ak.flatten(jet_matched_gen_vis_taus.eta, axis=None).compute())

        # Plot overall pt efficiency
        plt.clf()
        plot_efficiency(hist_eta_num, hist_eta_den)
        plt.title(f"(GenVisTau_eta_matched_all_jets/GenVisTau_eta)_{sample_name}")
        plt.savefig(os.path.join(eta_output_dir, f"eff_vs_eta_wo_iTLV_{sample_name}.pdf"))

        #############################################################################################################################################
        # Lxy efficency hists
        #############################################################################################################################################
        bins = np.arange(0, 15, 1)

        Lxy_axis = axis.Variable(bins, name="Lxy")
        hist_Lxy_den = Hist(Lxy_axis)
        hist_Lxy_num = Hist(Lxy_axis)
            
        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy, axis=None).compute())
        hist_Lxy_num.fill(ak.flatten(jet_matched_gen_vis_taus.parent.Lxy, axis=None).compute())

        # Plot overall pt efficiency
        plt.clf()
        plot_efficiency(hist_Lxy_num, hist_Lxy_den)
        plt.title(f"(GenVisTau_Lxy_matched_all_jets/GenVisTau_Lxy)_{sample_name}")
        plt.savefig(os.path.join(Lxy_output_dir, f"eff_vs_Lxy_wo_iTLV_{sample_name}.pdf"))
        
        #############################################################################################################################################
        # pT efficency hists
        #############################################################################################################################################
        pt_bins_low = np.arange(20, 101, 20)
        pt_bins_med = np.arange(100, 400, 30)
        pt_bins_high = np.arange(400, 600, 40)
        pt_bins_higher = np.arange(600, 1000, 50)
        pt_bins_eff = np.unique(np.concatenate([pt_bins_low, pt_bins_med, pt_bins_high, pt_bins_higher]))
        
        var_axes = {'pt': axis.Variable(pt_bins_eff, flow=False, name="GenVisTau_pt")}
        
        hist_pt_den = Hist(var_axes['pt'])
        hist_pt_num = Hist(var_axes['pt'])
            
        hist_pt_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.pt, axis=None).compute())
        hist_pt_num.fill(ak.flatten(jet_matched_gen_vis_taus.pt, axis=None).compute())
            
        # Plot overall pt efficiency
        plt.clf()
        plot_efficiency(hist_pt_num, hist_pt_den)
        plt.title(f"GenVisTau_pt_matched_all_jets/GenVisTau_pt_{sample_name}")
        plt.savefig(os.path.join(pT_output_dir, f"eff_vs_pt_wo_iTLV_{sample_name}.pdf"))
        
