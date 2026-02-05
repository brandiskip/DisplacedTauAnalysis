import ROOT
import math
import os
import awkward as ak
import vector
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

# --- Configuration ---
INPUT_FILE = "root://cmseos.fnal.gov///store/group/lpcdisptau//displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v14/LooseMuCosmic_2024/nano_1_0.root" 
OUTPUT_DIR = "event_displays"
MAX_EVENTS = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_cms_geometry(scope):
    """
    Draws CMS barrel layers based on official longitudinal engineering dimensions.
    Values are converted to cm (1 m = 100 cm).
    """
    # Radii based on the CMS Longitudinal View technical drawing:
    # 44.0 cm: Silicon Barrel (SB/1) inner boundary
    # 118.5 cm: Silicon Barrel (SB/1) outer boundary
    # 129.0 cm: Electromagnetic Barrel (EB/1) inner boundary
    # 181.1 cm: Hadron Barrel (HB/1) inner boundary
    # 295.0 cm: Solenoid Magnet inner wall
    # 380.0 cm: Solenoid Magnet outer wall
    # 402.0 cm: Muon Barrel Station 1 (MB/0/1) inner boundary
    # 490.5 cm: Muon Barrel Station 2 (MB/1/1) inner boundary
    # 597.5 cm: Muon Barrel Station 3 (MB/2/1) inner boundary
    # 700.0 cm: Muon Barrel Station 4 (MB/3/4) inner boundary
    
    radii = [44.0, 118.5, 129.0, 181.1, 295.0, 380.0, 402.0, 490.5, 597.5, 700.0]
    guides = []
    for r in radii:
        g = ROOT.TEllipse(0, 0, r, r)
        g.SetFillStyle(0)
        g.SetLineColor(ROOT.kGray)

        # Style logic: 
        # Dashed lines for internal tracker/calorimeter boundaries
        # Solid lines for the Solenoid and Muon Stations
        if r < 400:
            g.SetLineStyle(3) 
        else:
            g.SetLineStyle(1)
            g.SetLineWidth(2)
            
        g.Draw("same")
        guides.append(g)
    return guides

def run_event_display():
    # Load events
    print(f"Loading events from {INPUT_FILE}...")
    PFNanoAODSchema.mixins["DisMuon"] = "Muon"
    events = NanoEventsFactory.from_root(
        {INPUT_FILE: "Events"},
        schemaclass=PFNanoAODSchema
    ).events()

    # Filter for Cosmic-like events
    # Require at least 2 DisMuons AND at least 1 GenMuon
    gen_muons_all = events.GenPart[(abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1)]
    
    # Create a mask that ensures both collections have the required number of objects
    mask = (ak.num(events.DisMuon) >= 2) & (ak.num(gen_muons_all) >= 1)
    
    # Apply the mask to both events and the gen_muons collection simultaneously
    events = events[mask]
    gen_muons_filtered = gen_muons_all[mask]
    
    print(f"Found {len(events)} candidate events. Plotting first {MAX_EVENTS}...")

    # Lists to keep ROOT objects in memory
    arrows = []
    gen_arrows = []
    markers = []

    for i in range(min(MAX_EVENTS, len(events))):
        ev = events[i]
        muons = ev.DisMuon
        # Access gen muons for this specific event
        event_gen_muons = gen_muons_filtered[i]
        
        # Reco Muon Logic: sort by Phi to define the "Top" (+phi) and "Bottom" (-phi)
        sorted_idx = ak.argsort(muons.phi, ascending=False)
        m_sorted = muons[sorted_idx]
        
        # Calculate Delta T to determine if downward or upward
        # If delta_t < 0, the particle arrived at the top station first (Downward)
        t_top = m_sorted[0].timeAtIpInOut
        t_bottom = m_sorted[-1].timeAtIpInOut
        delta_t = t_top - t_bottom
        is_downward = delta_t < 0 

        # ROOT Canvas Setup
        canvas_name = f"run{ev.run}_evt{ev.event}"
        c = ROOT.TCanvas(canvas_name, canvas_name, 1000, 1000)
        scope = 800 
        frame = c.DrawFrame(-scope, -scope, scope, scope)
        frame.SetTitle(f"CMS Event Display: Reco vs Gen;x [cm];y [cm]")
        
        guides = draw_cms_geometry(scope)
        
        # Draw Gen Muons (Dashed Green)
        for gm in event_gen_muons:
            g_phi = gm.phi
            g_len = 700 # cm

            # Vector calculation: Gen arrows start at origin (0,0).
            # Projecting endpoints: x = r*cos(phi), y = r*sin(phi)
            gx1 = g_len * math.cos(g_phi)
            gy1 = g_len * math.sin(g_phi)
            
            g_arrow = ROOT.TArrow(0, 0, gx1, gy1, 0.02, "|>")
            g_arrow.SetLineColor(ROOT.kGreen + 2)
            g_arrow.SetLineStyle(7) # Dashed
            g_arrow.SetLineWidth(2)
            g_arrow.Draw()
            gen_arrows.append(g_arrow)

        # Draw Reco Muons (Solid Red/Blue)
        for j, m in enumerate(m_sorted):
            m_phi = m.phi
            # Calculate PCA using dxy and phi
            # Reco muons don't start at (0,0), they are displaced.
            # Using dxy, find the starting coordinate in XY:
            x0 = -m.dxy * math.sin(m_phi)
            y0 = m.dxy * math.cos(m_phi)
            
            # Draw PCA Marker
            pca_marker = ROOT.TMarker(x0, y0, 20)
            pca_marker.SetMarkerColor(ROOT.kBlack)
            pca_marker.SetMarkerSize(1.5)
            pca_marker.Draw()
            markers.append(pca_marker)

            # Draw Reco Arrows projecting direction
            length = 700 
            dir_mod = -1 if is_downward else 1

            # Final endpoint: start at PCA (x0, y0) and add the direction vector
            x1 = x0 + (dir_mod * length * math.cos(m_phi))
            y1 = y0 + (dir_mod * length * math.sin(m_phi))
            
            arrow = ROOT.TArrow(x0, y0, x1, y1, 0.02, "|>")
            arrow.SetLineWidth(3)
            arrow.SetLineColor(ROOT.kRed if j == 0 else ROOT.kBlue)
            arrow.Draw()
            arrows.append(arrow)

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.02)
        latex.DrawLatex(0.15, 0.88, f"Run: {ev.run}  Event: {ev.event}")
        latex.SetTextColor(ROOT.kGreen+2); latex.DrawLatex(0.15, 0.84, "Dashed Green: Gen Muon")
        latex.SetTextColor(ROOT.kRed);     latex.DrawLatex(0.15, 0.81, "Solid Red: Reco Upper Leg")
        latex.SetTextColor(ROOT.kBlue);    latex.DrawLatex(0.15, 0.78, "Solid Blue: Reco Lower Leg")

        # Save
        c.SaveAs(f"{OUTPUT_DIR}/{canvas_name}.png")
        print(f"Saved {canvas_name}.png")

if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True) 
    run_event_display()