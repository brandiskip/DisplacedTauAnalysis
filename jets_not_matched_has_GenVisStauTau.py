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
    'Stau_100_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
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

# ----------------------------------------------------------------------
# Main loop: Process each sample and produce histograms.
# ----------------------------------------------------------------------
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
        events['GenVisStauTaus'] = events.GenVisTau[(abs(events.GenVisTau.parent.pdgId) == 15) & \
                                                        (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) & \
                                                        (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) & \
                                                        (events.GenVisTau.parent.hasFlags("fromHardProcess")) & \
                                                        (events.GenVisTau.parent.Lxy < 100.0)]

        events = events[(ak.num(events.GenVisStauTaus) > 0)]

        events['GenMuon'] = gpart[(abs(gpart.pdgId) == 13) & (gpart.hasFlags("isLastCopy"))] 
        events.GenMuon = events.GenMuon[(events.GenMuon.pt > 20) & (abs(events.GenMuon.eta) < 2.4)]

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
        
        # add isTight to jets if lepton veto needed
        #jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTight) & (cut_filtered_events.Jet.chHEF > 0.01)]

        # add isTightLeptonVeto to jets if lepton veto needed
        jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTightLeptonVeto) & (cut_filtered_events.Jet.chHEF > 0.01)]
        #jets = jets[jets.disTauTag_score1 > 0.90]

        # Use these jet selections to trouble shoot looking for how each veto effects efficiency
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
        # Sort the selected jets by disTauTag_score1 (descending) and take the first jet per event
        sorted_by_score = jets[ak.argsort(jets.disTauTag_score1, ascending=False)]
        highest_score_jets = ak.singletons(ak.firsts(sorted_by_score))

        jets_matched = cut_filtered_events.GenVisStauTaus.nearest(highest_score_jets, threshold=0.4)
        jets_not_matched = highest_score_jets[delta_r_mask(highest_score_jets, cut_filtered_events.GenVisStauTaus,   0.4)]

        has_2_or_more_jets = ak.num(sorted_by_score) >= 2
        sorted_by_score_2j = sorted_by_score[has_2_or_more_jets]
        cut_filtered_events_2j = cut_filtered_events[has_2_or_more_jets]

        # Get the second highest scoring jet per event
        second_highest_score_jets = ak.singletons(sorted_by_score_2j[:, 1])
        highest_score_jets        = ak.singletons(sorted_by_score_2j[:, 0])

        # Match GenVisStauTaus to second highest scoring jets
        jets_matched_second_highest_score = cut_filtered_events_2j.GenVisStauTaus.nearest(second_highest_score_jets, threshold=0.4)

        # Select second-highest jets that were NOT matched
        #jets_not_matched_second_highest_score = second_highest_score_jets[delta_r_mask(second_highest_score_jets, cut_filtered_events.GenVisStauTaus, 0.4)]

        is_matched_to_second = ak.num(jets_matched_second_highest_score) > 0
        score_matched_2nd_jet = ak.flatten(second_highest_score_jets[is_matched_to_second].disTauTag_score1.compute())
        score_top_jet_in_matched_to_2nd = ak.flatten(highest_score_jets[is_matched_to_second].disTauTag_score1.compute())

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

        # Score of highest scoring jet not matched (no GenVisStauTau in event)
        scores_not_matched_top = ak.flatten(jets_not_matched.disTauTag_score1.compute())
        plt.hist(scores_not_matched_top, bins=bins, histtype='step', lw=2, label='Top Score (no match)', color='tab:orange')

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