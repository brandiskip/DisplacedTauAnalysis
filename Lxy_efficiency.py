import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import boost_histogram as bh 
import hist
import vector
from hist import Hist, axis, intervals
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import coffea.nanoevents.methods
import json
np.set_printoptions(precision=6, suppress=False, threshold=np.inf)

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
    #'Stau_100_1mm'   : 'root://cmseos.fnal.gov//store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_10mm'  : 'root://cmseos.fnal.gov//store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_100mm' : 'root://cmseos.fnal.gov//store/user/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_noskim_v1/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
}

PFNanoAODSchema.mixins["DisMuon"] = "Muon"
PFNanoAODSchema.error_missing_event_ids = False
samples = {}
#eventsnotselected = {}
for sample_name, files in filenames.items():
    samples[sample_name] = NanoEventsFactory.from_root(
        {files: "Events"},
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "MC"}
    ).events()
    #eventsnotselected[sample_name] = NanoEventsFactory.from_root(
        #{files: "EventsNotSelected"},
        #schemaclass=PFNanoAODSchema,
        #metadata={"dataset": "MC"}
    #).events()

def delta_r_mask(first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float) -> ak.highlevel.Array: 
    mval = first.metric_table(second) 
    return ak.all(mval > threshold, axis=-1)

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

output_dir = "Lxy_efficiency_plots"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    for sample_name in samples.keys():
        print(f"Processing sample: {sample_name}")
        events = samples[sample_name]

        # add dxy to jet fields
        charged_sel = events.Jet.constituents.pf.charge != 0
        dxy = ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0, axis = 2)
        events['Jet'] = ak.with_field(events.Jet, dxy, where="dxy")

        # add Lxy to GenVisTau.parent
        vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.parent.vx
        vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.parent.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        parent_with_Lxy = ak.with_field(events.GenVisTau.parent, Lxy, where="Lxy")
        events['GenVisTau'] = ak.with_field(events.GenVisTau, parent_with_Lxy, where="parent")

        events.Muon = events.Muon[(events.Muon.pt > 20) & (abs(events.Muon.eta) < 2.4) & (events.Muon.looseId == 1)]
        events.DisMuon = events.DisMuon[(events.DisMuon.pt > 20) & (abs(events.DisMuon.eta) < 2.4) & (events.DisMuon.looseId == 1)]
        events.Electron = events.Electron[(events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.4) & (events.Electron.convVeto)]
        events.Photon = events.Photon[(events.Photon.pt > 20) & (abs(events.Photon.eta) < 2.4) & (events.Photon.electronVeto)]

        ## find staus and their tau children
        gpart = events.GenPart
        events['staus'] = gpart[(abs(gpart.pdgId) == 1000015) & (gpart.hasFlags("isLastCopy"))] 

        events['staus_taus'] = events.staus.distinctChildren[ (abs(events.staus.distinctChildren.pdgId) == 15) & \
                                                          (events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                        (events.staus.distinctChildren.hasFlags("fromHardProcess")) \
                                                         ]
        events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & (events.GenVisTau.parent.hasFlags("fromHardProcess"))]

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

        # Select GenVisStauTaus with |eta| < 2.4 and pt > 20
        cut_filtered_events.GenVisStauTaus = cut_filtered_events.GenVisStauTaus[(cut_filtered_events.GenVisStauTaus.pt > 20) & (abs(cut_filtered_events.GenVisStauTaus.eta) < 2.4)]

        # Select jets with |eta| < 2.4 and pt > 20
        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
        #jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTightLeptonVeto)]
        
        GenVisTau_matched_to_jet = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        GenVisTau_matched_to_jet = ak.drop_none(GenVisTau_matched_to_jet)

        ################################################################################################################
        # Events Not Selected
        ################################################################################################################
        '''
        # for EventsNotSelected
        ens_events = eventsnotselected[sample_name]

        # Compute Lxy EventsNotSelected
        ens_Lxy = np.sqrt(ens_events.GenPart.vx**2 + ens_events.GenPart.vy**2)

        # Select staus from GenPart in eventsnotselected
        gpart_ens = ens_events.GenPart
        selected_staus = gpart_ens[(abs(gpart_ens.pdgId) == 1000015) & (gpart_ens.hasFlags(["fromHardProcess"])) & (gpart_ens.pt > 10)]

        ens_events['GenPart'] = ak.with_field(ens_events.GenPart, ens_Lxy, where="Lxy")

        ens_events['staus'] = gpart_ens[(abs(gpart_ens.pdgId) == 1000015) & (gpart_ens.hasFlags("isLastCopy"))] 

        ens_events['staus_taus'] = ens_events.staus.distinctChildren[ (abs(ens_events.staus.distinctChildren.pdgId) == 15) & \
                                                          (ens_events.staus.distinctChildren.hasFlags("isLastCopy")) & \
                                                        (ens_events.staus.distinctChildren.hasFlags("fromHardProcess")) \
                                                         ]

        ens_events['staus_taus'] = ak.firsts(ens_events.staus_taus[ak.argsort(ens_events.staus_taus.pt, ascending=False)], axis = 2)
        ens_staus_taus = ens_events['staus_taus']

        ens_mask_taul = ak.any((abs(ens_staus_taus.distinctChildren.pdgId) == 11) | (abs(ens_staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        ens_mask_tauh = ~ens_mask_taul

        ens_one_tauh_evt = (ak.sum(ens_mask_tauh, axis=-1) > 0) & (ak.sum(ens_mask_tauh, axis=-1) < 3)
        ens_one_taul_evt = (ak.sum(ens_mask_taul, axis=-1) > 0) & (ak.sum(ens_mask_taul, axis=-1) < 3)

        ens_filtered_events = ens_events[ens_one_tauh_evt & ens_one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau
    
        ens_tau_selections = ak.any((ens_filtered_events.staus_taus.pt > 20) & (abs(ens_filtered_events.staus_taus.eta) < 2.4), axis=-1)
        ens_num_taus = ak.num(ens_filtered_events.staus_taus[ens_tau_selections])
        ens_num_tau_mask = ens_num_taus > 1
        ens_cut_filtered_events = ens_filtered_events[ens_num_tau_mask]
        
        # Need to select visible decay products from events not selected
        lep_gen_taus = ak.any((abs(ens_cut_filtered_events.staus_taus.distinctChildren.pdgId) == 11) | (abs(ens_cut_filtered_events.staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        had_gen_taus = ~lep_gen_taus
        had_gen_taus = ens_cut_filtered_events.staus_taus[had_gen_taus]
        ens_gen_had_visibleChildren = had_gen_taus.distinctChildren[(abs(had_gen_taus.distinctChildren.pdgId) != 16)]
        ens_gen_had_visibleChildren = had_gen_taus.distinctChildren[ak.sum(ens_gen_had_visibleChildren.pt, axis=-1) > 10]
        '''
        ################################################################################################################
        # Plotting script
        ################################################################################################################
        '''
        Lxy_axis = axis.Regular(50, 0, 100, name="Lxy", label="Lxy [cm]")

        hist_Lxy_den = Hist(Lxy_axis)
        hist_Lxy_num = Hist(Lxy_axis)

        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy.compute(), axis=None))
        #hist_Lxy_den.fill(ak.flatten(ens_gen_had_visibleChildren.parent.Lxy.compute(), axis=None))
        hist_Lxy_num.fill(ak.flatten(GenVisTau_matched_to_jet.parent.Lxy.compute()))

        plt.clf()
        plot_efficiency(hist_Lxy_num, hist_Lxy_den)
        plt.title(f"Jet matched to GenVisStauTaus Efficiency vs Lxy: {sample_name}")
        plt.xlabel("Lxy [cm]")
        plt.ylabel("Efficiency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"eff_vs_Lxy_{sample_name}.pdf"))
        
        edges = [1e-4, 1e-3, 1e-2, 1, 5, 10, 15, 20, 25, 30, 40, 50, 100]

        Lxy_axis = axis.Variable(edges, name="Lxy", label="Lxy [cm]")

        hist_Lxy_den = Hist(Lxy_axis)
        hist_Lxy_num = Hist(Lxy_axis)

        hist_Lxy_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy.compute(), axis=None))
        hist_Lxy_num.fill(ak.flatten(GenVisTau_matched_to_jet.parent.Lxy.compute(), axis=None))

        plt.figure()
        plot_efficiency(hist_Lxy_num, hist_Lxy_den, log=True)
        plt.xlim(edges[0], edges[-1])          
        plt.ylim(0.0, 1.05)
        plt.xlabel("Lxy [cm]")
        plt.title(f"Jet–GenVisStauTaus efficiency vs Lxy : {sample_name}")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"eff_vs_Lxy_{sample_name}.pdf"))
        plt.close()
        '''
        display_edges = [
            1, 3, 6.5, 10, 15, 20, 30, 40, 50, 100       
        ]
        display_axis   = axis.Variable(display_edges, flow=False, name="Lxy", label="Lxy [cm]")
        den_display = Hist(display_axis)
        num_display = Hist(display_axis)

        den_display.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.parent.Lxy.compute(), axis=None))  
        num_display.fill(ak.flatten(GenVisTau_matched_to_jet.parent.Lxy.compute(), axis=None)) 

        fig, ax = plt.subplots(figsize=(6,4))

        num_display.plot1d(ax=ax, histtype="step", label="Numerator",   lw=1.5)
        den_display.plot1d(ax=ax, histtype="step", label="Denominator", lw=1.5)

        ax.set_xlabel("Lxy [cm]")
        ax.set_ylabel("Entries per bin")
        ax.set_title(f"Counts vs Lxy : {sample_name}")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"reg_counts_vs_Lxy_{sample_name}.pdf"))
        plt.close(fig)
        '''
        output_dir = "eta_efficiency_plots"
        eta_axis = axis.Regular(26, -2.5, 2.5, name="Lxy", label="Lxy [cm]")

        hist_eta_den = Hist(eta_axis)
        hist_eta_num = Hist(eta_axis)

        hist_eta_den.fill(ak.flatten(cut_filtered_events.GenVisStauTaus.eta.compute(), axis=None))
        hist_eta_num.fill(ak.flatten(GenVisTau_matched_to_jet.eta.compute()))

        plt.clf()
        plot_efficiency(hist_eta_num, hist_eta_den)
        plt.title(f"Jet matched to GenVisStauTaus Efficiency vs eta: {sample_name}")
        plt.xlabel("eta")
        plt.ylabel("Efficiency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"eff_vs_Lxy_{sample_name}.pdf"))

        mask    = cut_filtered_events.GenVisStauTaus.parent.Lxy > 1.0

        pt_arr  = ak.flatten(cut_filtered_events.GenVisStauTaus.pt[mask]).compute()
        eta_arr = ak.flatten(cut_filtered_events.GenVisStauTaus.eta[mask]).compute()
        plot_dir = "tau_Lxy1cm_plots"
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure()
        plt.hist(pt_arr, bins=np.linspace(0, 300, 61), histtype='step', lw=1.8)
        plt.xlabel(r"Gen $\tau_h$ $p_T$ [GeV]")
        plt.ylabel("Entries")
        plt.title(f"Gen τ$_h$ with Lxy > 1 cm: {sample_name}")
        plt.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"genTau_pt_LxyGT1cm_{sample_name}.pdf"))
        plt.close()

        plt.figure()
        plt.hist(eta_arr, bins=np.linspace(-2.5, 2.5, 26), histtype='step', lw=1.8)
        plt.xlabel(r"Gen $\tau_h$ $\eta$")
        plt.ylabel("Entries")
        plt.title(f"Gen τ$_h$ with Lxy > 1 cm: {sample_name}")
        plt.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"genTau_eta_LxyGT1cm_{sample_name}.pdf"))
        plt.close()
        '''