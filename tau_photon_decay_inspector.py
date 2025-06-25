import ROOT
import sys
from DataFormats.FWLite import Events, Handle
from math import *

def isAncestor(a, p):
    if a == p:
        return True
    for i in range(0, p.numberOfMothers()):
        if isAncestor(a, p.mother(i)):
            return True
    return False

# Sara's MiniAOD file
events = Events([
    'root://cms-xrd-global.cern.ch//store/mc/Run3Summer22EEMiniAODv4/SMS-TStauStau_MStau-100_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/50000/0b848c7b-1b00-4603-891a-6cd36af3d31b.root'
])

# Events to select (excluding the first two)
target_events = {
    (1, 353, 406311),
    (1, 402, 462667),
    (1, 439, 505068),
    (1, 441, 507889),
    (1, 451, 519874),
    (1, 515, 593504),
    (1, 167, 192514),
    (1, 184, 211491)
}

handlePruned = Handle("std::vector<reco::GenParticle>")
handlePacked = Handle("std::vector<pat::PackedGenParticle>")
labelPruned = ("prunedGenParticles")
labelPacked = ("packedGenParticles")

# Loop over events
for event in events:
    run = event.eventAuxiliary().run()
    lumi = event.eventAuxiliary().luminosityBlock()
    event_id = event.eventAuxiliary().event()

    if (run, lumi, event_id) not in target_events:
        continue

    print(f"\n==== Run: {run}, Lumi: {lumi}, Event: {event_id} ====")

    event.getByLabel(labelPacked, handlePacked)
    event.getByLabel(labelPruned, handlePruned)
    packed = handlePacked.product()
    pruned = handlePruned.product()

    for p in pruned:
        if abs(p.pdgId()) == 15:  # tau
            print(f"Tau: pdgId={p.pdgId()}, pt={p.pt():.2f}, eta={p.eta():.2f}, phi={p.phi():.2f}")
            print("  --> daughters:")
            for pa in packed:
                mother = pa.mother(0)
                if mother and isAncestor(p, mother):
                    print(f"      pdgId={pa.pdgId()}, pt={pa.pt():.2f}, eta={pa.eta():.2f}, phi={pa.phi():.2f}")
