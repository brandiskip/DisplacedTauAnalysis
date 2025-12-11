import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import hist
import vector
from hist import Hist, axis, intervals
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import coffea.nanoevents.methods
import json
np.set_printoptions(precision=6, suppress=False, threshold=np.inf)

# Load the file
filenames = {
    # Sara's sample
    #'Stau_100_0p01mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-0p01mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_100_0p1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-0p1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_100_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_200_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-200_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_0p01mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-0p01mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_300_0p1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-0p1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_300_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_0p01mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-0p01mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    #'Stau_500_0p1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-0p1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-10mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_100mm'  : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    'Stau_500_1000mm' : 'root://cmseos.fnal.gov///store/user/fiorendi/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/*.root',
    # Daniel's samples
    #'Stau_100_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_100_1mm/nano_0_44797_89593.root',
    #'Stau_100_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_100_10mm/nano_0_41936_83872.root',
    #'Stau_100_100mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_100_100mm/nano_0_0_46972.root',
    #'Stau_100_1000mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_100_1000mm/nano_0_58549_117098.root',
    #'Stau_200_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_200_1mm/nano_0_51035_102070.root',
    #'Stau_200_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_200_1mm/nano_0_102070_153104.root',
    #'Stau_200_10mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_200_10mm/nano_0_48361_96722.root',
    #'Stau_200_10mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_200_10mm/nano_0_96722_145082.root',
    #'Stau_200_100mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_200_100mm/nano_0_47878_95755.root',
    #'Stau_200_1000mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_200_1000mm/nano_0_47476_94952.root',
    #'Stau_200_1000mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_200_1000mm/nano_0_94952_142428.root',
    #'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_1mm/nano_0_100834_151251.root',
    #'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_1mm/nano_0_151251_201667.root',
    #'Stau_300_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_1mm/nano_0_50417_100834.root',
    #'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_10mm/nano_0_141522_188693.root',
    #'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_10mm/nano_0_47174_94348.root',
    #'Stau_300_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_10mm/nano_0_94348_141522.root',
    #'Stau_300_100mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_100mm/nano_0_44155_88310.root',
    #'Stau_300_100mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_100mm/nano_0_88310_132464.root',
    #'Stau_300_1000mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_1000mm/nano_0_102908_154362.root',
    #'Stau_300_1000mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_300_1000mm/nano_0_51454_102908.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_1mm/nano_0_141930_189240.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_1mm/nano_0_189240_236548.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_1mm/nano_0_47310_94620.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_1mm/nano_0_94620_141930.root',
    #'Stau_500_1mm'    : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_1mm/nano_1_0_8918.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_10mm/nano_0_139713_186284.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_10mm/nano_0_186284_232853.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_10mm/nano_0_46571_93142.root',
    #'Stau_500_10mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_10mm/nano_0_93142_139713.root',
    #'Stau_500_100mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_100mm/nano_0_135876_181168.root',
    #'Stau_500_100mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_100mm/nano_0_45292_90584.root',
    #'Stau_500_100mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_100mm/nano_0_90584_135876.root',
    #'Stau_500_1000mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_1000mm/nano_0_115650_173474.root',
    #'Stau_500_1000mm'   : 'root://cmseos.fnal.gov///store/group/lpcdisptau/dally/displacedTaus/skim/Summer22_CHS_v10/mutau/v2/Stau_500_1000mm/nano_0_57825_115650.root',  
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

#out_dir_numjets     = "jet_multiplicity_plots"
out_dir_genmuon     = "deltaR_GenMuon_plots"
#out_dir_genvistau   = "deltaR_GenVisStauTau_plots"

#os.makedirs(out_dir_numjets, exist_ok=True)
os.makedirs(out_dir_genmuon, exist_ok=True)
#os.makedirs(out_dir_genvistau, exist_ok=True)

#os.makedirs("plots/jetID_vars", exist_ok=True)
os.makedirs("plots/matched_jet_scores", exist_ok=True)

deltaR_overlap_dict = {}
deltaR_dict = {}

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
        dxy_err = abs(ak.flatten(events.Jet.constituents.pf[ak.argmax(events.Jet.constituents.pf[charged_sel].pt, axis=2, keepdims=True)].d0Err, axis = 2))
        #events['Jet'] = ak.with_field(events.Jet, dxy_err, where="dxy_err")
        events['Jet'] = ak.with_field(events.Jet, ak.zeros_like(events.Jet.pt) - 999.0, where="dxy_err")

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
        
        vx = events.GenVisTau.parent.vx - events.GenVisTau.parent.distinctParent.vx
        vy = events.GenVisTau.parent.vy - events.GenVisTau.parent.distinctParent.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        parent_with_Lxy = ak.with_field(events.GenVisTau.parent, Lxy, where="Lxy")
        events['GenVisTau'] = ak.with_field(events.GenVisTau, parent_with_Lxy, where="parent")

        '''
        events['Muon'] = events.Muon[(events.Muon.pt > 20) & (abs(events.Muon.eta) < 2.4) & (events.Muon.looseId == 1)]
        events['DisMuon'] = events.DisMuon[(events.DisMuon.pt > 20) & (abs(events.DisMuon.eta) < 2.4) & (events.DisMuon.looseId == 1)]
        events['Electron'] = events.Electron[(events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.4) & (events.Electron.convVeto)]
        events['Photon'] = events.Photon[(events.Photon.pt > 20) & (abs(events.Photon.eta) < 2.4) & (events.Photon.electronVeto)]
        '''

        ## find staus and their tau children
        gpart = events.GenPart
        events['staus'] = events.GenPart[(abs(events.GenPart.pdgId) == 1000015) & (events.GenPart.hasFlags("isLastCopy"))] 

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
        #print(f"GenVisStauTaus pt: {events.GenVisStauTaus.pt.compute()}")

        #events = events[(ak.num(events.GenVisStauTaus) > 0)]
        '''                                           
        events['GenMuon'] = events.GenPart[(abs(events.GenPart.pdgId) == 13) & (events.GenPart.hasFlags("isLastCopy"))] 
        #print(f"Before Selections GenMuon pt: {events.GenMuon.pt.compute()}")
        vx = events.GenMuon.vx
        vy = events.GenMuon.vy
        Lxy = np.sqrt(vx**2 + vy**2)
        events['GenMuon'] = ak.with_field(events.GenMuon, Lxy, where="Lxy")

        events['GenMuon'] = events.GenMuon[(events.GenMuon.pt > 20) & \
                                            (abs(events.GenMuon.eta) < 2.4) & \
                                            (abs(events.GenMuon.distinctParent.distinctParent.pdgId) == 1000015)]
        '''
        '''
        mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenMuon) == 1)
        events = events[mask]
        mask_good_events = ((events.run == 1) &
                            (events.luminosityBlock == 59) &
                            (events.event == 82194))
        mask_bad_events = ((events.run == 1) &
                            (events.luminosityBlock == 59) &
                            (events.event == 82195))
        
        if ak.any(mask_good_events).compute():
            print("GenMuon pt:", ak.to_list(events.GenMuon.pt[mask_good_events].compute()))
            print("GenMuon parent pt:", ak.to_list(events.GenMuon.distinctParent.pt[mask_good_events].compute()))
            print("GenMuon grandparent pt:", ak.to_list(events.GenMuon.distinctParent.distinctParent.pt[mask_good_events].compute()))

        if ak.any(mask_bad_events).compute():    
            print(f"After pt, eta, parent selections GenMuon pt: {events.GenMuon.pt[mask_bad_events].compute()}")
            print(f"GenMuon parent pt: {events.GenMuon.distinctParent.pt[mask_bad_events].compute()}")
            print(f"GenMuon grandparent pt: {events.GenMuon.distinctParent.distinctParent.pt[mask_bad_events].compute()}")
        '''
        #events = events[(ak.num(events.GenMuon) > 0)]
        '''
        events['GenElectron'] = events.GenPart[(abs(events.GenPart.pdgId) == 11) & (events.GenPart.hasFlags("isLastCopy"))]
        vx = events.GenElectron.vx
        vy = events.GenElectron.vy
        electron_Lxy = np.sqrt(vx**2 + vy**2)
        events['GenElectron'] = ak.with_field(events.GenElectron, electron_Lxy, where="Lxy") 
        events['GenElectron'] = events.GenElectron[(events.GenElectron.pt > 20) & \
                                                    (abs(events.GenElectron.eta) < 2.4) & \
                                                    (events.GenElectron.Lxy < 100.0) & \
                                                    (abs(events.GenElectron.distinctParent.distinctParent.pdgId) == 1000015)]
        
        mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenMuon) == 1) & (ak.num(events.GenElectron) == 0)
        events = events[mask]
        '''
        '''
        mask = (ak.num(events.GenVisStauTaus) == 1) & (ak.num(events.GenElectron) == 1) & (ak.num(events.GenMuon) == 0)
        events = events[mask]
        '''

        #events['GenJet'] = events.GenJet[(events.GenJet.pt > 20) & (abs(events.GenJet.eta) < 2.4)]

        # Use for Mykyta plots "other jets from signal" and "other jets from PU"
        # https://indico.cern.ch/event/1451074/contributions/6108905/attachments/2942042/5169340/score_2017.pdf
        '''
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.GenVisStauTaus, 0.5)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.GenElectron, 0.5)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.GenMuon, 0.5)]
        events['Jet'] = events.Jet[delta_r_mask(events.Jet, events.GenJet, 0.5)]
        '''
        
        #print("Stau tau pt at event 9400", events.staus_taus.pt.compute()[9400])
        #arg_sort = ak.argsort(events.staus_taus.pt, ascending=False).compute()
        #print("arg sort at event 9400", arg_sort[9400].compute())
        '''
        arg_sort = ak.argsort(events.staus_taus.pt, ascending=False)
        bad_mask = ak.to_numpy(ak.any(arg_sort == -6, axis=-1).compute())
        bad_idx  = np.flatnonzero(bad_mask)

        runs  = np.asarray(events.run.compute())[bad_idx]
        lumis = np.asarray(events.luminosityBlock.compute())[bad_idx]
        evts  = np.asarray(events.event.compute())[bad_idx]

        for i, r, l, e in zip(bad_idx, runs, lumis, evts):
            print(f"Bad index -6 at event_idx={int(i)} run={int(r)} lumi={int(l)} event={int(e)}")
        '''

        '''
        for i in range(len(arg_sort)):
             if -6 in arg_sort[i]:
                 print(f"Event {i}, {arg_sort[i]}, has a problem with indices")
        '''
        
        events['staus_taus'] = ak.firsts(events.staus_taus[ak.argsort(events.staus_taus.pt, ascending=False)], axis = 2)
        staus_taus = events['staus_taus']
        # print("gvt",events.GenVisStauTaus.pt.compute() )
        # print("Jets", events.Jet.pt.compute())
        mask_taul = ak.any((abs(staus_taus.distinctChildren.pdgId) == 11) | (abs(staus_taus.distinctChildren.pdgId) == 13), axis=-1)
        mask_tauh = ~mask_taul

        one_tauh_evt = (ak.sum(mask_tauh, axis=-1) > 0) & (ak.sum(mask_tauh, axis=-1) < 3)
        one_taul_evt = (ak.sum(mask_taul, axis=-1) > 0) & (ak.sum(mask_taul, axis=-1) < 3)

        filtered_events = events[one_tauh_evt & one_taul_evt]  # Filtered events are events with one hadronic tau and one leptonic tau
    
        tau_selections = ak.any((filtered_events.staus_taus.pt > 20) & (abs(filtered_events.staus_taus.eta) < 2.4), axis=-1)
        num_taus = ak.num(filtered_events.staus_taus[tau_selections])
        num_tau_mask = num_taus > 1
        cut_filtered_events = filtered_events[(num_tau_mask)]

        jets_all = cut_filtered_events.Jet

        '''
        # leading charged PF candidate per jet
        pf_all = jets_all.constituents.pf
        charged_pf_all = pf_all[pf_all.charge != 0]
        sorted_by_p_all = charged_pf_all[ak.argsort(charged_pf_all.p, ascending=False)]
        lead_pf_all = ak.firsts(sorted_by_p_all)

        # hcalFraction for the leading PF cand
        lead_hcal = lead_pf_all.hcalFraction
        lead_hcal_valid = ~ak.is_none(lead_hcal)

        # jet-level mask: leading PF exists and has hcalFraction > 0.2
        hcal_lead_gt_02_per_jet = lead_hcal_valid & (lead_hcal > 0.2)
        '''

        # Select GenVisStauTaus with |eta| < 2.4 and pt > 20
        #cut_filtered_events.GenVisStauTaus = cut_filtered_events.GenVisStauTaus[(cut_filtered_events.GenVisStauTaus.pt > 20) & (abs(cut_filtered_events.GenVisStauTaus.eta) < 2.4)]

        #jets = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]
        
        # Select jets with |eta| < 2.4 and pt > 20
        # Use these jet selections to trouble shoot looking for how each veto effects efficiency
        #jets_all = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20)]

        # add isTight to jets
        #jets_tight = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & (cut_filtered_events.Jet.pt > 20) & (cut_filtered_events.Jet.isTight)]
    
        # add isTightLeptonVeto to jets
        base_jet_mask = (
            (abs(cut_filtered_events.Jet.eta) < 2.4) &
            (cut_filtered_events.Jet.pt > 20) &
            (cut_filtered_events.Jet.neHEF < 0.99) &
            (cut_filtered_events.Jet.neEmEF < 0.9) &
            ((cut_filtered_events.Jet.chMultiplicity + cut_filtered_events.Jet.neMultiplicity) > 1) &
            (cut_filtered_events.Jet.chMultiplicity > 0) &
            (cut_filtered_events.Jet.muEF < 0.5) &
            (cut_filtered_events.Jet.chEmEF < 0.8)
        )

        #jets_tightLeptonVeto = cut_filtered_events.Jet[base_jet_mask & hcal_lead_gt_02_per_jet]

        jets_tightLeptonVeto = cut_filtered_events.Jet[base_jet_mask]
        '''
        jets_tightLeptonVeto = cut_filtered_events.Jet[(abs(cut_filtered_events.Jet.eta) < 2.4) & \
                                (cut_filtered_events.Jet.pt > 20) & \
                                (cut_filtered_events.Jet.isTightLeptonVeto)]
        '''
        '''
        if sample_name in ["Stau_300_100mm"]:
            deltaR_matrix = cut_filtered_events.GenVisStauTaus.metric_table(cut_filtered_events.GenMuon).compute()
            deltaR_flat = ak.ravel(deltaR_matrix)
            
            label = sample_name.replace("Stau_", "m=").replace("_100mm", " GeV")
            deltaR_dict[label] = deltaR_flat

        if deltaR_dict:
            plt.figure()
            bins = np.arange(0, 5.05, 0.05)

            for label, deltaR_array in deltaR_dict.items():
                plt.hist(deltaR_array, bins=bins, histtype='step', lw=2, label=label)

        plt.xlabel(r'$\Delta R$(GenVisStauTau, GenMuon)')
        plt.ylabel("Number of pairs")
        plt.title(r'$\Delta R$ between GenVisStauTau and GenMuon')
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/deltaR_GenVisTau_GenMuon_100mm_samples.pdf")
        plt.close()
        '''

        '''
        # ΔR to GenMuon
        deltaR_muon_all = jets_all.metric_table(cut_filtered_events.GenMuon).compute()
        deltaR_muon_tight = jets_tight.metric_table(cut_filtered_events.GenMuon).compute()
        deltaR_muon_tightLeptonVeto = jets_tightLeptonVeto.metric_table(cut_filtered_events.GenMuon).compute()

        # ΔR to GenVisStauTau
        deltaR_vis_all = jets_all.metric_table(cut_filtered_events.GenVisStauTaus).compute()
        deltaR_vis_tight = jets_tight.metric_table(cut_filtered_events.GenVisStauTaus).compute()
        deltaR_vis_tightLeptonVeto = jets_tightLeptonVeto.metric_table(cut_filtered_events.GenVisStauTaus).compute()

        def extract_matched_dRs(dr_muon, dr_vis):
            # Flatten both arrays
            dr_muon_flat = ak.flatten(dr_muon, axis=None)
            dr_vis_flat = ak.flatten(dr_vis, axis=None)

            x = ak.to_numpy(dr_muon_flat)
            y = ak.to_numpy(dr_vis_flat)

            return x, y

        # Apply to each jet category
        x_all, y_all = extract_matched_dRs(deltaR_muon_all, deltaR_vis_all)
        x_tight, y_tight = extract_matched_dRs(deltaR_muon_tight, deltaR_vis_tight)
        x_tightLeptonVeto, y_tightLeptonVeto = extract_matched_dRs(deltaR_muon_tightLeptonVeto, deltaR_vis_tightLeptonVeto)

        bins = np.linspace(0, 3.5, 101)

        # Plotting function
        def plot_2D(x, y, title, fname):
            plt.figure()
            plt.hist2d(x, y, bins=[bins, bins], cmap='viridis', norm=LogNorm())
            plt.xlabel(r'$\Delta R$(jet, GenMuon)')
            plt.ylabel(r'$\Delta R$(jet, GenVisStauTau)')
            plt.title(title)
            plt.colorbar(label="Counts")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_genmuon, f"{fname}_{sample_name}.pdf"))
            plt.close()

        # Generate all 2D plots
        plot_2D(x_all, y_all, r'2D $\Delta R$: All jets', "deltaR2D_AllJets")
        plot_2D(x_tight, y_tight, r'2D $\Delta R$: isTight jets', "deltaR2D_TightJets")
        plot_2D(x_tightLeptonVeto, y_tightLeptonVeto, r'2D $\Delta R$: isTightLeptonVeto jets', "deltaR2D_TightLeptonVetoJets")
        '''
        
        '''
        jet_id_vars = {
            "Jet_muEF": jets_all.muEF,
            "Jet_chEmEF": jets_all.chEmEF,
            "Jet_chHEF": jets_all.chHEF,
            "Jet_neHEF": jets_all.neHEF,
            "Jet_neMultiplicity": jets_all.neMultiplicity,
            "Jet_sumMultiplicity": jets_all.chMultiplicity + jets_all.neMultiplicity,
        }

        tight_vars = {
            "Jet_muEF": jets_tight.muEF,
            "Jet_chEmEF": jets_tight.chEmEF,
            "Jet_chHEF": jets_tight.chHEF,
            "Jet_neHEF": jets_tight.neHEF,
            "Jet_neMultiplicity": jets_tight.neMultiplicity,
            "Jet_sumMultiplicity": jets_tight.chMultiplicity + jets_tight.neMultiplicity,
        }

        tightlepveto_vars = {
            "Jet_muEF": jets_tightLeptonVeto.muEF,
            "Jet_chEmEF": jets_tightLeptonVeto.chEmEF,
            "Jet_chHEF": jets_tightLeptonVeto.chHEF,
            "Jet_neHEF": jets_tightLeptonVeto.neHEF,
            "Jet_neMultiplicity": jets_tightLeptonVeto.neMultiplicity,
            "Jet_sumMultiplicity": jets_tightLeptonVeto.chMultiplicity + jets_tightLeptonVeto.neMultiplicity,
        }

        for var_name in jet_id_vars:
            plt.figure()

            if var_name == "Jet_muEF":
                bins = np.linspace(0, 1.0, 101)  # bin width 0.01

            elif var_name == "Jet_chEmEF":
                bins = np.linspace(0, 1.0, 101)  # bin width 0.01

            elif var_name == "Jet_chHEF":
                bins = np.linspace(0, 0.05, 51)  # bin width 0.001

            elif var_name == "Jet_neHEF":
                bins = np.linspace(0, 1.2, 121)  # bin width 0.01

            elif var_name == "Jet_neMultiplicity":
                bins = np.arange(-0.5, 21.5, 1)  # centers on integers

            elif var_name == "Jet_sumMultiplicity":
                bins = np.arange(-0.5, 41.5, 1)  # bin width 1, centered on integers

            else:
                bins = np.linspace(0, 1.0, 101)

            plt.hist(ak.ravel(jet_id_vars[var_name]).compute(), bins=bins, histtype='step', label='All jets', lw=2)
            plt.hist(ak.ravel(tight_vars[var_name]).compute(), bins=bins, histtype='step', label='isTight', lw=2)
            plt.hist(ak.ravel(tightlepveto_vars[var_name]).compute(), bins=bins, histtype='step', label='isTightLeptonVeto', lw=2)

            plt.xlabel(var_name)
            plt.ylabel("Number of jets")
            plt.title(f"{var_name} for Different Jet ID Criteria")
            plt.legend()
            plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"plots/jetID_vars/{var_name}_{sample_name}.pdf")
            plt.close()
            '''
        '''
        ###################################################################################################
        # Plots for each type of jetId
        ###################################################################################################
        num_jets_all = ak.num(jets_all).compute()
        num_jets_tight = ak.num(jets_tight).compute()
        num_jets_tightLeptonVeto = ak.num(jets_tightLeptonVeto).compute()

        max_jets = max(
            ak.max(num_jets_all, initial=0),
            ak.max(num_jets_tight, initial=0),
            ak.max(num_jets_tightLeptonVeto, initial=0)
        )

        bins = np.arange(0, max_jets + 2)

        plt.figure()
        plt.hist(num_jets_all, bins=bins, histtype='step', lw=2, label='All jets')
        plt.hist(num_jets_tight, bins=bins, histtype='step', lw=2, label='isTight')
        plt.hist(num_jets_tightLeptonVeto, bins=bins, histtype='step', lw=2, label='isTightLeptonVeto')
        plt.xlabel("Number of jets per event")
        plt.ylabel("Number of events")
        plt.title(f"Jet Multiplicity: {sample_name}")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_numjets, f"numjets_{sample_name}.pdf"))
        plt.close()
        '''
        ###################################################################################################
        # Plots for deltaR for GenMuon wrt jets
        ###################################################################################################
        '''
        deltaR_all = jets_all.metric_table(cut_filtered_events.GenMuon).compute()
        deltaR_tight = jets_tight.metric_table(cut_filtered_events.GenMuon).compute()
        deltaR_tightLeptonVeto = jets_tightLeptonVeto.metric_table(cut_filtered_events.GenMuon).compute()
        
        #plt.figure()
        bins = np.linspace(0, 5, 50)

        
        plt.hist(ak.ravel(deltaR_all), bins=bins, histtype='step', lw=2, label='All jets')
        plt.hist(ak.ravel(deltaR_tight), bins=bins, histtype='step', lw=2, label='isTight')
        plt.hist(ak.ravel(deltaR_tightLeptonVeto), bins=bins, histtype='step', lw=2, label='isTightLeptonVeto')

        plt.xlabel(r'$\Delta R$(jet, GenMuon)')
        plt.ylabel("Number of jet-muon pairs")
        plt.title(r'$\Delta R$ between jets w/score > 0.9 (jets_chHEF > 0.01) and GenMuons')
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_genmuon, f"deltaR_GenMuon_{sample_name}.pdf"))
        plt.close()
        '''
        ###################################################################################################
        # Plots for deltaR for GenVisStauTau wrt jets
        ###################################################################################################
        '''
        deltaR_vis_all = jets_all.metric_table(cut_filtered_events.GenVisStauTaus).compute()
        deltaR_vis_tight = jets_tight.metric_table(cut_filtered_events.GenVisStauTaus).compute()
        deltaR_vis_tightLeptonVeto = jets_tightLeptonVeto.metric_table(cut_filtered_events.GenVisStauTaus).compute()

        deltaR_vis_all = ak.flatten(deltaR_vis_all, axis=-1)
        deltaR_vis_tight = ak.flatten(deltaR_vis_tight, axis=-1)
        deltaR_vis_tightLeptonVeto = ak.flatten(deltaR_vis_tightLeptonVeto, axis=-1)

        deltaR_vis_all = ak.broadcast_arrays(deltaR_vis_all, deltaR_all)
        deltaR_vis_tight = ak.broadcast_arrays(deltaR_vis_tight, deltaR_tight)
        deltaR_vis_tightLeptonVeto = ak.broadcast_arrays(deltaR_vis_tightLeptonVeto, deltaR_tightLeptonVeto)

        plt.figure()
        plt.hist2d(ak.ravel(deltaR_all), ak.ravel(deltaR_vis_all), bins=[bins, bins], cmap='viridis')
        plt.xlabel(r'$\Delta R$(jet, GenMuon)')
        plt.ylabel(r'$\Delta R$(jet, GenVisStauTau)')
        plt.title(r'2D $\Delta R$: All jets')
        plt.colorbar(label="Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_genmuon, f"deltaR2D_AllJets_{sample_name}.pdf"))
        plt.close()

        plt.figure()
        plt.hist2d(ak.ravel(deltaR_tight), ak.ravel(deltaR_vis_tight), bins=[bins, bins], cmap='viridis')
        plt.xlabel(r'$\Delta R$(jet, GenMuon)')
        plt.ylabel(r'$\Delta R$(jet, GenVisStauTau)')
        plt.title(r'2D $\Delta R$: isTight jets')
        plt.colorbar(label="Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_genmuon, f"deltaR2D_TightJets_{sample_name}.pdf"))
        plt.close()

        plt.figure()
        plt.hist2d(ak.ravel(deltaR_tightLeptonVeto), ak.ravel(deltaR_vis_tightLeptonVeto), bins=[bins, bins], cmap='viridis')
        plt.xlabel(r'$\Delta R$(jet, GenMuon)')
        plt.ylabel(r'$\Delta R$(jet, GenVisStauTau)')
        plt.title(r'2D $\Delta R$: isTightLeptonVeto jets')
        plt.colorbar(label="Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_genmuon, f"deltaR2D_TightLeptonVetoJets_{sample_name}.pdf"))
        plt.close()
        
        # Plot
        plt.figure()
        bins = np.linspace(0, 5, 50)

        plt.hist(ak.ravel(deltaR_vis_all), bins=bins, histtype='step', lw=2, label='All jets')
        plt.hist(ak.ravel(deltaR_vis_tight), bins=bins, histtype='step', lw=2, label='isTight')
        plt.hist(ak.ravel(deltaR_vis_tightLeptonVeto), bins=bins, histtype='step', lw=2, label='isTightLeptonVeto')

        plt.xlabel(r'$\Delta R$(jet, GenVisStauTau)')
        plt.ylabel("Number of jet-tau pairs")
        plt.title(r'$\Delta R$ between jets and GenVisStauTaus')
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_genvistau, f"deltaR_GenVisTau_{sample_name}.pdf"))
        plt.close()
        '''
        '''
        jet_matched_gen_vis_taus = cut_filtered_events.GenVisStauTaus.nearest(jets_tightLeptonVeto, threshold=0.4)
        jet_matched_gen_vis_taus = ak.drop_none(jet_matched_gen_vis_taus)
        nMatched_jets_matched_to_gen_vis_tau_all_jets = ak.sum(ak.num(jet_matched_gen_vis_taus))
        '''
        '''
        #jet_matched_gen_vis_taus = jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        #jet_matched_gen_vis_taus = ak.drop_none(jet_matched_gen_vis_taus)

        #jet_matched_gen_vis_taus_score = cut_filtered_events.GenVisStauTaus.nearest(highest_score_jets, threshold=0.4)
        #jet_matched_gen_vis_taus_score = ak.drop_none(jet_matched_gen_vis_taus_score)

        # Sort the selected jets by pt (descending) and take the first jet per event
        #sorted_by_pt = jets[ak.argsort(jets.pt, ascending=False)]
        #leading_pt_jets = ak.singletons(ak.firsts(sorted_by_pt))

        # Sort the selected jets by dxy (descending) and take the first jet per event
        #sorted_by_dxy = jets[ak.argsort(abs(jets.dxy), ascending=False)]
        #highest_dxy_jets = ak.singletons(ak.firsts(sorted_by_dxy))
        '''
        # Make cut on highest score jets
        #jets_tightLeptonVeto = jets_tightLeptonVeto[jets_tightLeptonVeto.disTauTag_score1 > 0.90]

        #sorted_by_dxy_err = jets_tightLeptonVeto[ak.argsort(jets_tightLeptonVeto.dxy_err, ascending=True)]
        #lowest_dxy_err = ak.singletons(ak.firsts(sorted_by_dxy_err))
        '''
        sorted_by_pt = jets_tightLeptonVeto[ak.argsort(jets_tightLeptonVeto.pt, ascending=False)]
        leading_pt_jets = ak.singletons(ak.firsts(sorted_by_pt))
        '''
        # Sort the selected jets by disTauTag_score1 (descending) and take the first jet per event
        sorted_by_score = jets_tightLeptonVeto[ak.argsort(jets_tightLeptonVeto.disTauTag_score1, ascending=False)]
        highest_score_jets = ak.singletons(ak.firsts(sorted_by_score))
        
        ##########################################################################################################
        # GenVisTau matched_leading_jets
        ##########################################################################################################
        num_vis_gen_taus = ak.sum(ak.num(cut_filtered_events.GenVisStauTaus))
        

        '''
        gen_vis_taus_matched_by_pt = leading_pt_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        gen_vis_taus_matched_by_pt = ak.drop_none(gen_vis_taus_matched_by_pt)
        nMatched_gen_vis_taus_highest_pt_jet = ak.sum(ak.num(gen_vis_taus_matched_by_pt))

        # Compute pt efficiency
        efficiency = (nMatched_gen_vis_taus_highest_pt_jet / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0
        '''
        '''
        # Matching using the dxy leading jets
        gen_vis_taus_matched_highest_dxy_jet = highest_dxy_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        gen_vis_taus_matched_highest_dxy_jet = ak.drop_none(gen_vis_taus_matched_highest_dxy_jet)
        nMatched_gen_vis_taus_highest_dxy_jet = ak.sum(ak.num(gen_vis_taus_matched_highest_dxy_jet))
        
        # Compute dxy efficiency
        efficiency = (nMatched_gen_vis_taus_highest_dxy_jet / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0
        '''
        
        # Matching using the leading-score jets.
        gen_vis_taus_matched_by_score = highest_score_jets.nearest(cut_filtered_events.GenVisStauTaus, threshold=0.4)
        gen_vis_taus_matched_by_score = ak.drop_none(gen_vis_taus_matched_by_score)
        jet_matched_gen_vis_taus_score = cut_filtered_events.GenVisStauTaus.nearest(highest_score_jets, threshold=0.4)
        jet_matched_gen_vis_taus_score = ak.drop_none(jet_matched_gen_vis_taus_score)
        nMatched_jets_matched_to_gen_vis_tau_highest_score_jet = ak.sum(ak.num(jet_matched_gen_vis_taus_score))
        
        
        '''
        #########################################################################################################
        # Block of code to re-create Mykyta's plots for study of tagger score
        #########################################################################################################
        jet_matched_gen_vis_taus_score = cut_filtered_events.GenVisStauTaus.nearest(jets, threshold=0.3)
        jet_matched_gen_vis_taus_score = ak.drop_none(jet_matched_gen_vis_taus_score)

        jet_matched_gen_muon_score = cut_filtered_events.GenMuon.nearest(jets, threshold=0.3)
        jet_matched_gen_muon_score = ak.drop_none(jet_matched_gen_muon_score)
        #jet_matched_gen_muon_score = jet_matched_gen_muon_score[(jet_matched_gen_muon_score.disTauTag_score1 > 0.15) & (jet_matched_gen_muon_score.disTauTag_score1 < 0.25)]
        deltaR_GenMuon = jet_matched_gen_muon_score.metric_table(cut_filtered_events.GenMuon)

        plt.figure()
        plt.hist(ak.ravel(deltaR_GenMuon).compute(), bins=np.linspace(0, 0.4, 41), histtype='step', lw=2, label='GenMuon')
        plt.xlabel("ΔR(GenMuon, Jet)")
        plt.ylabel("Number of matched muon-jet pairs")
        plt.yscale("log")
        #plt.title(f"ΔR: GenMuon-Jet, 0.15 < score < 0.25 ({sample_name})")
        plt.title(f"ΔR: GenMuon-Jet ({sample_name})")
        plt.grid(True, ls='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/matched_jet_scores/deltaR_muon_score_bin_{sample_name}.pdf")
        plt.close()

        jet_matched_gen_electron_score = cut_filtered_events.GenElectron.nearest(jets, threshold=0.3)
        jet_matched_gen_electron_score = ak.drop_none(jet_matched_gen_electron_score)
        #jet_matched_gen_electron_score = jet_matched_gen_electron_score[(jet_matched_gen_electron_score.disTauTag_score1 > 0.15) & (jet_matched_gen_electron_score.disTauTag_score1 < 0.25)]
        deltaR_GenElectron = jet_matched_gen_electron_score.metric_table(cut_filtered_events.GenElectron)

        plt.figure()
        plt.hist(ak.ravel(deltaR_GenElectron).compute(), bins=np.linspace(0, 0.4, 41), histtype='step', lw=2, label='GenElectron')
        plt.xlabel("ΔR(GenElectron, Jet)")
        plt.ylabel("Number of matched electron-jet pairs")
        plt.yscale("log")
        #plt.title(f"ΔR: GenElectron-Jet, 0.15 < score < 0.25 ({sample_name})")
        plt.title(f"ΔR: GenElectron-Jet ({sample_name})")
        plt.grid(True, ls='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/matched_jet_scores/deltaR_electron_score_bin_{sample_name}.pdf")
        plt.close()
        
        jet_matched_GenJet = cut_filtered_events.GenJet.nearest(jets, threshold=0.3)

        bins = np.linspace(0, 1.0, 101)

        plt.figure()
        #plt.hist(ak.ravel(jet_matched_gen_vis_taus_score.disTauTag_score1).compute(), bins=bins, histtype='step', lw=2, label='GenVisTau')
        #plt.hist(ak.ravel(jet_matched_gen_muon_score.disTauTag_score1).compute(), bins=bins, histtype='step', lw=2, label='GenMuon')
        #plt.hist(ak.ravel(jet_matched_gen_electron_score.disTauTag_score1).compute(), bins=bins, histtype='step', lw=2, label='GenElectron')
        #plt.hist(ak.ravel(jet_matched_GenJet.disTauTag_score1).compute(), bins=bins, histtype='step', lw=2, label='GenJet')
        plt.hist(ak.ravel(jets.disTauTag_score1).compute(), bins=bins, histtype='step', lw=2, label='GenJet')
        plt.xlabel("Jet Score")
        plt.ylabel("Number of matched jets")
        plt.title(f"Jet Score for Remaining Jets - {sample_name}")
        plt.yscale("log")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/matched_jet_scores/jet_scores_remaining_jets_{sample_name}.pdf")
        plt.close()
        '''

        
        # Compute score efficiency
        efficiency = (nMatched_jets_matched_to_gen_vis_tau_highest_score_jet / num_vis_gen_taus).compute() if num_vis_gen_taus.compute() > 0 else 0.0
      
        # Get mass and lifetime from sample_name (e.g., "Stau_100_1mm")
        parts = sample_name.split('_')
        mass = int(parts[1]) 
        lifetime = int(parts[2].replace('mm', '')) # remove mm from 1mm 

        # Create a dictionary with keys: mass, lifetime, efficiency
        efficiency_data = {
            "mass": mass,
            "lifetime": lifetime,
            "efficiency": efficiency
        }

        # Save to JSON file
        json_filename = "jets_highest_score_efficiency_results_isTightLV.json"

        # Ensure JSON file exists and is not empty before loading
        if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
            try:
                with open(json_filename, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {json_filename} is corrupted. Overwriting with new data.")
                existing_data = []  # Reset the JSON file if it's corrupted
        else:
            print(f"Creating new JSON file: {json_filename}")
            existing_data = []  # If the file doesn't exist or is empty, initialize as an empty list

        existing_data.append(efficiency_data)

        with open(json_filename, "w") as f:
            json.dump(existing_data, f, indent=4)

    # Read the JSON file
    with open("jets_highest_score_efficiency_results_isTightLV.json", "r") as f:
        efficiency_data = json.load(f)

    # Extract unique masses and lifetimes
    masses = sorted(set(entry["mass"] for entry in efficiency_data))
    lifetimes = sorted(set(entry["lifetime"] for entry in efficiency_data))  # Still in mm

    # Create a mapping from mass/lifetime to an array index
    mass_idx = {m: i for i, m in enumerate(masses)}
    lifetime_idx = {lt: i for i, lt in enumerate(lifetimes)}

    # Build efficiency grid (rows=lifetimes, columns=masses)
    Z = np.zeros((len(lifetimes), len(masses)))

    for entry in efficiency_data:
        m = entry["mass"]
        lt = entry["lifetime"]
        Z[lifetime_idx[lt], mass_idx[m]] = entry["efficiency"]

    # Reverse the y-axis order so lifetimes go from smallest to largest
    Z = Z[::-1]
    lifetimes = lifetimes[::-1]

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the grid as an image
    cmap = cm.get_cmap("plasma")  # Use a color map where efficiency determines shade
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Efficiency is in range [0,1]
    im = ax.imshow(Z, cmap=cmap, norm=norm)

    # Set x-axis (Mass) and y-axis (Lifetime)
    ax.set_xticks(range(len(masses)))
    ax.set_xticklabels([str(m) for m in masses])  # Mass values as labels

    ax.set_yticks(range(len(lifetimes)))
    ax.set_yticklabels([f"{lt} mm" for lt in lifetimes])  # Lifetimes in mm
    
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel(r"$c\tau$ [mm]")
    plt.title("(nMatched_gen_vis_taus_highest_score_jet)/(num_vis_gen_taus)[d, $a_{vis , j}, L, isTLV$]", fontsize=10, pad=15)

    # Loop over data dimensions and create text annotations.
    for i in range(len(lifetimes)):
        for j in range(len(masses)):
            efficiency_value = Z[i, j]
            if efficiency_value > 0:  # Only display values where efficiency is nonzero
                text_color = "white" if efficiency_value < 0.5 else "black"
                ax.text(j, i, f"{efficiency_value:.3f}", ha="center", va="center", 
                        color=text_color, fontsize=9)

    # Add colorbar to indicate efficiency scale
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Efficiency")
    output_file = "jets_highest_score_efficiency_results_isTightLV.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved efficiency plot to {output_file}")