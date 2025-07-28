Folder contents:
This directory contains histograms of selected jet variables for unmatched jets (i.e. jets not within ΔR < 0.4 of any GenVisStauTau) in events that contain at least one GenVisStauTau.

Histogram Contents:
The following histograms are created per sample:

JetPt_NotMatched.pdf — transverse momentum of not matched jets

JetEta_NotMatched.pdf — pseudorapidity of not matched jets

JetDxy_NotMatched.pdf — transverse impact parameter of not matched jets

JetScore_NotMatched.pdf — disTauTag score 1 of not matched jets

Event Selections:
GenVisStauTau Selection:

Must come from a tau (|pdgId| = 15) that came from a stau (|pdgId| = 1000015)

Tau must be:
- fromHardProcess
- Descendant of a stau that is isLastCopy

Transverse decay distance of parent tau:
- Lxy < 100 cm

Kinematic cuts on visible tau decay:
- pt > 20 GeV
- |eta| < 2.4

Event-level Filter:
- Must contain exactly one hadronic tau and one leptonic tau (from Gen tau decay tree)
- Both must pass:
  - pt > 20 GeV
  - |eta| < 2.4

Jets must satisfy:
- pt > 20 GeV
- |eta| < 2.4
- isTight = True
- chargedHadronEnergyFraction (chHEF) > 0.01

One jet per event is selected:
- The highest disTauTag_score1 jet in the event

Matching to GenVisStauTau:
- A jet is matched if ΔR(jet, GenVisStauTau) < 0.4
- A jet is not matched if ΔR > 0.4 for all GenVisStauTaus in the event

Conditions for Plots in This Folder:
Only events with:
- ≥ 1 GenVisStauTau
- ≥ 1 unmatched jet

Only these unmatched jets are used in the histograms
