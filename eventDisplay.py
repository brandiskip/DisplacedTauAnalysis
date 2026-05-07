import ROOT
import math
import os
import awkward as ak
import vector
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

# --- Configuration ---
INPUT_FILE = "root://cmseos.fnal.gov///store/group/lpcdisptau//displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v15/LooseMuCosmic_2024/nano_2_0.root" 
OUTPUT_DIR = "event_displays"
MAX_EVENTS = 100
DRAW_RZ = True
os.makedirs(OUTPUT_DIR, exist_ok=True)

vector.register_awkward()

def draw_cms_geometry(scope):
    """
    Draws CMS barrel layers based.
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

def draw_cms_geometry_rz():
    """
    Draws a simplified CMS Longitudinal (R-Z) view.
    Horizontal axis is z (beamline), vertical axis is y.
    """
    guides = []
    # Barrel DTs (Horizontal lines)
    radii = [402.0, 490.5, 597.5, 700.0]
    z_barrel = 660.0
    for r in radii:
        for sign in [1, -1]:
            l = ROOT.TLine(-z_barrel, sign * r, z_barrel, sign * r)
            l.SetLineColor(ROOT.kGray)
            l.SetLineStyle(1)
            l.SetLineWidth(2)
            l.Draw("same")
            guides.append(l)
            
    # Endcap CSCs (Vertical lines)
    z_csc = [600.0, 700.0, 800.0, 900.0, 1000.0]
    r_min, r_max = 100.0, 700.0
    for z in z_csc:
        for sign_z in [1, -1]:
            for sign_r in [1, -1]:
                l = ROOT.TLine(sign_z * z, sign_r * r_min, sign_z * z, sign_r * r_max)
                l.SetLineColor(ROOT.kGray)
                l.SetLineStyle(1)
                l.SetLineWidth(2)
                l.Draw("same")
                guides.append(l)
    return guides

def run_event_display():
    # Load events
    print(f"Loading events from {INPUT_FILE}...")
    PFNanoAODSchema.mixins["DisMuon"] = "Muon"
    events = NanoEventsFactory.from_root(
        {INPUT_FILE: "Events"},
        schemaclass=PFNanoAODSchema
    ).events()

    events["DisMuon"] = ak.zip(
        {
            "pt": events.DisMuon.pt,
            "ptErr": events.DisMuon.ptErr,
            "eta": events.DisMuon.eta,
            "phi": events.DisMuon.phi,
            "mass": events.DisMuon.mass,
            "charge": events.DisMuon.charge,
            "timeAtIpInOut": events.DisMuon.timeAtIpInOut,
            "timeAtIpInOutErr": events.DisMuon.timeAtIpInOutErr,
            "timeNDof": events.DisMuon.timeNDof,
            "dxy": events.DisMuon.dxy,
            "numberOfValidMuonDTHits": events.DisMuon.numberOfValidMuonDTHits,
            "numberOfValidMuonCSCHits": events.DisMuon.numberOfValidMuonCSCHits,
            "numberOfValidMuonHits": events.DisMuon.numberOfValidMuonHits,
            "dtStationsWithValidHits": events.DisMuon.dtStationsWithValidHits,
            "eta_at_ecal": events.DisMuon.eta_at_ecal,
            "phi_at_ecal": events.DisMuon.phi_at_ecal,
            "eta_at_mb2": events.DisMuon.eta_at_mb2,
            "phi_at_mb2": events.DisMuon.phi_at_mb2,
            "isGlobal": events.DisMuon.isGlobal, 
            "isStandalone": events.DisMuon.isStandalone,
            "staTrackNormChi2": events.DisMuon.staTrackNormChi2,
            "mediumId": events.DisMuon.mediumId,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=events.behavior,
    )

    events["GenPart"] = ak.zip(
        {
            "pt": events.GenPart.pt,
            "eta": events.GenPart.eta,
            "phi": events.GenPart.phi,
            "mass": events.GenPart.mass,
            "pdgId": events.GenPart.pdgId,
            "status": events.GenPart.status,
            "vx": events.GenPart.vx,
            "vy": events.GenPart.vy,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=events.behavior,
    )

    dis_muons = events.DisMuon

    # Standalone Requirements
    sta_hit_base = (dis_muons.numberOfValidMuonCSCHits + dis_muons.numberOfValidMuonDTHits) > 12
    sta_hit_csc0 = ak.where(dis_muons.numberOfValidMuonCSCHits == 0, dis_muons.numberOfValidMuonDTHits > 18, True)
    sta_chi2 = dis_muons.staTrackNormChi2 < 2.5
    sta_pterr = (dis_muons.ptErr / dis_muons.pt) < 1.0
    
    mask_sta = dis_muons.isStandalone & sta_hit_base & sta_hit_csc0 & sta_chi2 & sta_pterr

    # Global Requirements
    glb_hit = (dis_muons.numberOfValidMuonCSCHits + dis_muons.numberOfValidMuonDTHits) > 12
    glb_pterr = (dis_muons.ptErr / dis_muons.pt) < 0.3
    
    mask_glb = dis_muons.isGlobal & glb_hit & glb_pterr

    good_muon_mask = mask_sta | mask_glb
    events["DisMuon"] = events.DisMuon[good_muon_mask]
    dis_muons = dis_muons[good_muon_mask]

    # Filter for Cosmic-like events
    # Require at least 2 DisMuons AND at least 1 GenMuon
    gen_muons = events.GenPart[(abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1)]
    
    mask_basic = (ak.num(dis_muons) >= 2) & (ak.num(gen_muons) >= 1)
    
    events = events[mask_basic]
    gen_muons = gen_muons[mask_basic]
    dis_muons = dis_muons[mask_basic]

    # This is for removing "duplicate" tracks
    #######################################################################################################
    sorted_pt = events.DisMuon[ak.argsort(events.DisMuon.pt, axis=1, ascending=False)]
    lead = sorted_pt[:, 0]
    sublead = sorted_pt[:, 1]

    deta = lead.eta - sublead.eta
    dphi = lead.delta_phi(sublead)
    dpt = lead.pt - sublead.pt
    
    mask_same_charge = (lead.charge * sublead.charge) > 0
    is_duplicate = mask_same_charge & (abs(deta) < 0.01) & (abs(dphi) < 0.001) & (abs(dpt) < 0.5)
    
    events = events[~is_duplicate]
    gen_muons = gen_muons[~is_duplicate]
    #######################################################################################################

    dis_muons = events.DisMuon
    upper_candidates = dis_muons[dis_muons.phi > 0]
    lower_candidates = dis_muons[dis_muons.phi < 0]

    n_upper = ak.num(upper_candidates)
    n_lower = ak.num(lower_candidates)
    n_total = ak.num(dis_muons)

    has_both_legs = (n_upper >= 1) & (n_lower >= 1)
        
    events = events[has_both_legs]
    gen_muons = gen_muons[has_both_legs]
    dis_muons = dis_muons[has_both_legs]

    lead = lead[has_both_legs]          
    sublead = sublead[has_both_legs]

    upper_candidates = upper_candidates[has_both_legs]
    lower_candidates = lower_candidates[has_both_legs]

    '''
    # If >1 candidate per side, keep the one with the highest pt
    # Sort descending by pt and take the first index ([:, 0])
    upper_candidates = upper_candidates[ak.argsort(upper_candidates.pt, axis=1, ascending=False)]
    lower_candidates = lower_candidates[ak.argsort(lower_candidates.pt, axis=1, ascending=False)]
    '''

    # Have exactly one Upper and one Lower muon per valid event
    upper = upper_candidates[:, 0]
    lower = lower_candidates[:, 0]

    ndof_quality = (upper.timeNDof > 7) & (lower.timeNDof > 7)

    events = events[ndof_quality]
    gen_muons = gen_muons[ndof_quality]
    dis_muons = dis_muons[ndof_quality]

    lead = lead[ndof_quality]          
    sublead = sublead[ndof_quality]

    upper = upper[ndof_quality]
    lower = lower[ndof_quality]

    delta_t = (upper.timeAtIpInOut - lower.timeAtIpInOut)

    dot_prod = upper.px * lower.px + upper.py * lower.py + upper.pz * lower.pz
    denom = upper.p * lower.p
    cosA = ak.where(denom != 0, dot_prod / denom, -1000.0)

    mask_lt_50 = ak.fill_none(upper.timeAtIpInOut < -50, False)
    mask_not_b2b = ak.fill_none(cosA >= -0.99, False)
    mask_zero_dt = ak.fill_none(upper.numberOfValidMuonDTHits == 0, False)

    mask_final = mask_lt_50 & mask_not_b2b & mask_zero_dt

    events = events[mask_final]
    gen_muons = gen_muons[mask_final]
    dis_muons = dis_muons[mask_final]

    upper = upper[mask_final]
    lower = lower[mask_final]

    delta_t = delta_t[mask_final]
    cosA = cosA[mask_final]

    '''
    mask_lt_50 = ak.fill_none(upper.timeAtIpInOut < -50, False)
    mask_lt_50_not_b2b = mask_lt_50 & ak.fill_none(cosA >= -0.99, False)

    events = events[mask_lt_50_not_b2b]
    gen_muons = gen_muons[mask_lt_50_not_b2b]
    dis_muons = dis_muons[mask_lt_50_not_b2b]

    upper = upper[mask_lt_50_not_b2b]
    lower = lower[mask_lt_50_not_b2b]

    delta_t = delta_t[mask_lt_50_not_b2b]
    cosA = cosA[mask_lt_50_not_b2b]
    '''

    '''
    upper_time_mask = upper.timeAtIpInOut < -50.0
    cosA_mask = cosA >= -0.99

    events = events[upper_time_mask & cosA_mask]
    gen_muons = gen_muons[upper_time_mask & cosA_mask]
    dis_muons = dis_muons[upper_time_mask & cosA_mask]

    lead = lead[upper_time_mask & cosA_mask]          
    sublead = sublead[upper_time_mask & cosA_mask]

    upper = upper[upper_time_mask & cosA_mask]
    lower = lower[upper_time_mask & cosA_mask]

    delta_t = delta_t[upper_time_mask & cosA_mask]
    cosA = cosA[upper_time_mask & cosA_mask]
    '''

    '''
    # Lower phi Muon must have phi > 0
    mask_pos_phi = lower.phi > 0

    events = events[mask_pos_phi]
    gen_muons = gen_muons[mask_pos_phi]
    '''

    # Define the Back-to-Back Mask
    #mask_back_to_back = cosA < -0.99
    #mask_duplicates = cosA > 0.99

    # Apply Final Filtering
    #events = events[mask_back_to_back]
    #gen_muons = gen_muons[mask_back_to_back]

    #events = events[mask_duplicates]
    #gen_muons = gen_muons[mask_duplicates]
    
    #print(f"Events passing opposite direction cut (cosA < -0.99): {len(events)}")
    #print(f"Events passing opposite direction cut (cosA > 0.99): {len(events)}")
    #print(f"Events passing Final Cuts (Lower.phi > 0): {len(events)}")
    #print(f"Events that have duplicate tracks: {len(events)}")
    print(f"Plotting first {MAX_EVENTS}...")

    for i in range(min(MAX_EVENTS, len(events))):
        # Initialize lists to keep objects alive
        arrows = []
        gen_arrows = []
        markers = []
        latex_objects = []

        ev = events[i]
        upper_mu = upper[i]
        lower_mu = lower[i]
        
        event_gen_muons = gen_muons[i]
        leading_gen = event_gen_muons[0]

        # Extract scalar values for this specific event
        evt_delta_t = delta_t[i] 
        evt_cosA = cosA[i]  
        
        dr_upper = upper_mu.delta_r(leading_gen)
        dr_lower = lower_mu.delta_r(leading_gen)

        # Setup Canvas
        canvas_name = f"run{ev.run}_evt{ev.event}"
        c = ROOT.TCanvas(canvas_name, canvas_name, 1000, 1000)
        scope = 800 
        frame = c.DrawFrame(-scope, -scope, scope, scope)
        frame.SetTitle(f"Upper Timing Studies: Run {ev.run} Evt {ev.event};x [cm];y [cm]")

        # Draw Geometry
        geometry_guides = draw_cms_geometry(scope) 
        
        # Draw Gen Muons
        for gm in event_gen_muons:
            gx0 = gm.vx 
            gy0 = gm.vy
            
            # Calculate end point based on direction
            gx1 = gx0 + 1500 * math.cos(gm.phi)
            gy1 = gy0 + 1500 * math.sin(gm.phi)
            
            g_arrow = ROOT.TArrow(gx0, gy0, gx1, gy1, 0.02, "|>")
            g_arrow.SetLineColor(ROOT.kGreen + 2)
            g_arrow.SetLineStyle(7)
            g_arrow.SetLineWidth(2)
            g_arrow.Draw()
            gen_arrows.append(g_arrow)

        '''
        # Draw Reco Muons (Upper to Red, Lower to Blue)
        reco_muons_to_draw = [(upper_mu, ROOT.kRed), (lower_mu, ROOT.kBlue)]
        
        for m, color in reco_muons_to_draw:
            m_phi = m.phi
            x0 = -m.dxy * math.sin(m_phi)
            y0 = m.dxy * math.cos(m_phi)
            
            # Draw Raw Vector
            length = 700 
            x1 = x0 + (length * math.cos(m_phi))
            y1 = y0 + (length * math.sin(m_phi))
            
            arrow = ROOT.TArrow(x0, y0, x1, y1, 0.02, "|>")
            arrow.SetLineWidth(3)
            arrow.SetLineColor(color)
            arrow.Draw()
            arrows.append(arrow)

            # Draw PCA Marker
            pca_marker = ROOT.TMarker(x0, y0, 20)
            pca_marker.SetMarkerColor(ROOT.kBlack)
            pca_marker.SetMarkerSize(1.5)
            pca_marker.Draw()
            markers.append(pca_marker)
        '''
        event_all_muons = ev.DisMuon 
        
        for m in event_all_muons:
            color = ROOT.kRed if m.phi > 0 else ROOT.kBlue
            m_phi = m.phi
            x0 = -m.dxy * math.sin(m_phi)
            y0 = m.dxy * math.cos(m_phi)
            
            # Draw Raw Vector
            length = 700 
            x1 = x0 + (length * math.cos(m_phi))
            y1 = y0 + (length * math.sin(m_phi))
            
            arrow = ROOT.TArrow(x0, y0, x1, y1, 0.02, "|>")
            arrow.SetLineWidth(3)
            arrow.SetLineColor(color)
            arrow.Draw()
            arrows.append(arrow)

            # Draw PCA Marker
            pca_marker = ROOT.TMarker(x0, y0, 20)
            pca_marker.SetMarkerColor(ROOT.kBlack)
            pca_marker.SetMarkerSize(1.5)
            pca_marker.Draw()
            markers.append(pca_marker)

        # --- LEGEND ---
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.016) 

        current_y = 0.88
        
        # Header
        latex.SetTextColor(ROOT.kBlack)
        #latex.DrawLatex(0.12, current_y, f"Run: {ev.run}  Event: {ev.event}")
        latex.DrawLatex(0.12, current_y, f"Run: {ev.run}  Event: {ev.event} (Total Reco Muons: {len(event_all_muons)})")
        current_y -= 0.03

        # Gen Muon Stats
        latex.SetTextColor(ROOT.kGreen+2)
        latex.DrawLatex(0.12, current_y, f"Gen Muon: pT={leading_gen.pt:.1f}, #eta={leading_gen.eta:.2f}, #phi={leading_gen.phi:.2f}")
        current_y -= 0.03
        
        # Upper Stats (Red)
        latex.SetTextColor(ROOT.kRed)
        latex.DrawLatex(0.12, current_y, f"Muon 1 (Upper): t={upper_mu.timeAtIpInOut:.1f}ns, #DeltaR={dr_upper:.3f}")
        current_y -= 0.02
        latex.DrawLatex(0.12, current_y, f"                pT={upper_mu.pt:.1f}, #eta={upper_mu.eta:.2f}, #phi={upper_mu.phi:.2f}")
        current_y -= 0.03
        
        # Lower Stats (Blue)
        latex.SetTextColor(ROOT.kBlue)
        latex.DrawLatex(0.12, current_y, f"Muon 2 (Lower): t={lower_mu.timeAtIpInOut:.1f}ns, #DeltaR={dr_lower:.3f}")
        current_y -= 0.02
        latex.DrawLatex(0.12, current_y, f"                pT={lower_mu.pt:.1f}, #eta={lower_mu.eta:.2f}, #phi={lower_mu.phi:.2f}")
        current_y -= 0.03 

        # Delta T and Cos Alpha
        latex.SetTextColor(ROOT.kBlack)
        latex.DrawLatex(0.12, current_y, f"#Delta t: {evt_delta_t:.2f} ns")
        current_y -= 0.03
        latex.DrawLatex(0.12, current_y, f"cos(#alpha): {evt_cosA:.3f}") 

        c.SaveAs(f"{OUTPUT_DIR}/{canvas_name}.png")
        print(f"Saved {canvas_name}.png")

    
if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True) 
    run_event_display()