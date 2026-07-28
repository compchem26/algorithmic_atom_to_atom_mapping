import io
import streamlit as st
import networkx as nx
import matplotlib
import textwrap
matplotlib.use('Agg') # Ensures Matplotlib runs safely in a web thread without GUI popups
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import ListedColormap
from rdkit import Chem
from rdkit.Chem import rdDepictor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="RDKit Hybrid Atom-to-Atom Mapper", layout="wide")

# ==========================================
# HELPERS & CONVERTERS
# ==========================================
def format_mapping(mapping_dict, chunk_size=6):
    if not mapping_dict:
        return "{ }"
    items = [f"{k}$_r$:{v}$_p$" for k, v in mapping_dict.items()]
    lines = [", ".join(items[i:i+chunk_size]) for i in range(0, len(items), chunk_size)]
    return "{ " + "\n  ".join(lines) + " }"

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Invalid SMILES string: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    
    G = nx.Graph()
    pos = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        G.add_node(idx, type=symbol)
        coords = mol.GetConformer().GetAtomPosition(idx)
        pos[idx] = (coords.x, coords.y)
        
    for bond in mol.GetBonds():
        order = bond.GetBondTypeAsDouble()
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=order)
        
    return G, pos

# ==========================================
# VISUALIZER
# ==========================================
class ReactionVisualizer:
    def __init__(self, g1, g2, pos1, pos2):
        self.g1 = g1
        self.g2 = g2
        self.nodes_1 = list(g1.nodes())
        self.nodes_2 = list(g2.nodes())
        
        self.pos1 = pos1
        self.pos2 = pos2
        self.pos1_id = {k: (v[0] + 0.2, v[1] + 0.2) for k, v in self.pos1.items()}
        self.pos2_id = {k: (v[0] + 0.2, v[1] + 0.2) for k, v in self.pos2.items()}
        
        self.history = []
        self.cmap = ListedColormap(['#f8f9fa', '#74b9ff', '#ffeaa7', '#ff7675', '#00b894', '#00cec9'])
        
        # 👇 CHANGE THIS SECTION
        # Use a wider figure to comfortably fit 3 items side-by-side
        self.fig = plt.figure(figsize=(20, 8), facecolor='#ffffff')
        
        # Create a 2-row, 3-column grid. 
        # height_ratios=[3, 1] gives the visual plots more vertical space than the text
        gs = self.fig.add_gridspec(2, 3, height_ratios=[3, 1], wspace=0.25, hspace=0.05)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])     # Top Left: Reactants
        self.ax2 = self.fig.add_subplot(gs[0, 1])     # Top Middle: Products
        self.ax3 = self.fig.add_subplot(gs[0, 2])     # Top Right: Selection Matrix
        
        # The colon ':' makes the info text span entirely across the bottom row
        self.ax_info = self.fig.add_subplot(gs[1, :])

    def record_state(self, phase_name, status, mapping, global_best_info="", candidate_n=None, candidate_m=None, locked_scaffold=None, broken_bonds=None, formed_bonds=None):
        self.history.append({
            'phase_name': phase_name,
            'status': status,
            'global_best_info': global_best_info,
            'mapping': mapping.copy() if mapping else {},
            'candidate_n': candidate_n,
            'candidate_m': candidate_m,
            'locked_scaffold': locked_scaffold.copy() if locked_scaffold else {},
            'broken_bonds': broken_bonds.copy() if broken_bonds else [],
            'formed_bonds': formed_bonds.copy() if formed_bonds else []
        })

    def _fig_to_pil(self):
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=90)
        buf.seek(0)
        return Image.open(buf)

    def _draw_chemical_bonds(self, G, pos, ax, highlight_edges=None, highlight_color='#ff7675', full_mapping=None, target_G=None, is_g1=True):
        highlight_edges = highlight_edges or []
        h_edges = {tuple(sorted((u, v))) for u, v in highlight_edges}
        
        for u, v, d in G.edges(data=True):
            order = d.get('order', 1.0)
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length == 0: continue
            
            nx_vec, ny_vec = -dy/length, dx/length
            offset = 0.12  
            
            edge_key = tuple(sorted((u, v)))
            is_highlighted = edge_key in h_edges
            
            changed_order = order
            if is_highlighted and full_mapping and target_G:
                if is_g1:
                    m_u, m_v = full_mapping.get(u), full_mapping.get(v)
                else:
                    inv_map = {val: key for key, val in full_mapping.items()}
                    m_u, m_v = inv_map.get(u), inv_map.get(v)
                    
                if m_u is not None and m_v is not None and target_G.has_edge(m_u, m_v):
                    target_order = target_G[m_u][m_v]['order']
                    changed_order = abs(order - target_order)
                else:
                    changed_order = order 
            elif is_highlighted:
                changed_order = order

            def draw_line(xs, xe, ys, ye, is_changed):
                color = highlight_color if is_changed else '#888888'
                ls = '--' if is_changed else '-'
                lw = 3 if is_changed else 2
                z = 2 if is_changed else 1
                ax.plot([xs, xe], [ys, ye], color=color, lw=lw, ls=ls, zorder=z)
            
            if order == 1.0: 
                draw_line(x1, x2, y1, y2, is_highlighted)
            elif order == 2.0: 
                draw_line(x1 + nx_vec*offset, x2 + nx_vec*offset, y1 + ny_vec*offset, y2 + ny_vec*offset, changed_order >= 1 and is_highlighted)
                draw_line(x1 - nx_vec*offset, x2 - nx_vec*offset, y1 - ny_vec*offset, y2 - ny_vec*offset, changed_order >= 2 and is_highlighted)
            elif order == 3.0: 
                draw_line(x1, x2, y1, y2, changed_order >= 1 and is_highlighted)
                draw_line(x1 + nx_vec*offset*1.5, x2 + nx_vec*offset*1.5, y1 + ny_vec*offset*1.5, y2 + ny_vec*offset*1.5, changed_order >= 2 and is_highlighted)
                draw_line(x1 - nx_vec*offset*1.5, x2 - nx_vec*offset*1.5, y1 - ny_vec*offset*1.5, y2 - ny_vec*offset*1.5, changed_order >= 3 and is_highlighted)
            elif order == 1.5: 
                draw_line(x1, x2, y1, y2, changed_order >= 1 and is_highlighted)
                c = highlight_color if (changed_order >= 1.5 and is_highlighted) else '#888888'
                lw = 3 if (changed_order >= 1.5 and is_highlighted) else 2
                z = 2 if (changed_order >= 1.5 and is_highlighted) else 1
                ax.plot([x1 + nx_vec*offset, x2 + nx_vec*offset], [y1 + ny_vec*offset, y2 + ny_vec*offset], color=c, lw=lw, ls='--', zorder=z)
            else: 
                draw_line(x1, x2, y1, y2, is_highlighted)

    def get_preview_frame(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax_info.clear()

        # self.fig.suptitle("Preview Mode", color='#2c3e50', fontweight="bold", fontsize=18, x=0.05, ha='left', y=0.98)
        
        # # Explicitly push the grid down so it doesn't overlap the title
        # self.fig.subplots_adjust(top=0.85)
        self.ax1.set_title("Reactants (G1)", color='#2c3e50', fontweight="bold", fontsize=14, pad=15)
        self.ax2.set_title("Products (G2)", color='#2c3e50', fontweight="bold", fontsize=14, pad=15)
        self.ax3.set_title("Selection Matrix", color='#2c3e50', fontweight="bold", fontsize=14, pad=15)
        
        self.ax_info.axis('off')
        self.ax_info.text(0.5, 0.5, "Awaiting Algorithm Run...", color='#95a5a6', fontsize=14, style='italic', ha='center', va='center')
        
        nx.draw_networkx_nodes(self.g1, self.pos1, ax=self.ax1, node_color='#f0f3f5', node_size=800, edgecolors='#cbd5e1', linewidths=2)
        nx.draw_networkx_nodes(self.g2, self.pos2, ax=self.ax2, node_color='#f0f3f5', node_size=800, edgecolors='#cbd5e1', linewidths=2)
        
        self._draw_chemical_bonds(self.g1, self.pos1, self.ax1)
        self._draw_chemical_bonds(self.g2, self.pos2, self.ax2)
        
        type_labels_1 = {n: self.g1.nodes[n].get('type', '') for n in self.nodes_1}
        type_labels_2 = {m: self.g2.nodes[m].get('type', '') for m in self.nodes_2}
        nx.draw_networkx_labels(self.g1, self.pos1, labels=type_labels_1, ax=self.ax1, font_size=12, font_weight='bold', font_color='black')
        nx.draw_networkx_labels(self.g2, self.pos2, labels=type_labels_2, ax=self.ax2, font_size=12, font_weight='bold', font_color='black')

        id_labels_1 = {n: f"{n}$_r$" for n in self.nodes_1}
        id_labels_2 = {m: f"{m}$_p$" for m in self.nodes_2}
        nx.draw_networkx_labels(self.g1, self.pos1_id, labels=id_labels_1, ax=self.ax1, font_size=10, font_weight='bold', font_color='black')
        nx.draw_networkx_labels(self.g2, self.pos2_id, labels=id_labels_2, ax=self.ax2, font_size=10, font_weight='bold', font_color='black')
        
        self.ax1.set_aspect('equal')
        self.ax2.set_aspect('equal')
        self.ax1.margins(0.2)
        self.ax2.margins(0.2)
        self.ax3.axis('off')
        
        return self._fig_to_pil()

    def get_frame(self, index):
        if not self.history:
            return self.get_preview_frame()

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax_info.clear()
        self.ax3.axis('on') 
        self.ax_info.axis('off')
        
        state = self.history[index]
        phase_name = state['phase_name']
        status = state['status']
        global_best = state['global_best_info']
        mapping = state['mapping']
        candidate_n = state['candidate_n']
        candidate_m = state['candidate_m']
        locked_scaffold = state['locked_scaffold']
        broken_bonds = state['broken_bonds']
        formed_bonds = state['formed_bonds']

        is_rejection = "Rejected" in status or "Fail" in status
        wrapped_status = textwrap.fill(status, width=80)

        # title_color = '#e74c3c' if is_rejection else '#2c3e50'
        # self.fig.suptitle(f"{phase_name}\n{wrapped_status}", color=title_color, fontweight="bold", fontsize=16, x=0.05, ha='left', y=0.98)
        # self.fig.subplots_adjust(top=0.80)

        self.ax1.set_title(f"Reactants (G1)", color='#2c3e50', fontweight="bold", fontsize=14, pad=15)
        self.ax2.set_title(f"Products (G2)", color='#2c3e50', fontweight="bold", fontsize=14, pad=15)
        self.ax3.set_title("Selection Matrix", color='#2c3e50', fontweight="bold", fontsize=14, pad=15)
        
        info_text = f"Step {index}/{len(self.history)-1}\n\n{global_best}"
        self.ax_info.text(0.5, 0.5, info_text, color='#34495e', fontweight="bold", fontsize=12, 
                          ha='center', va='center', bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=1.2'))

        # --- MATRIX ---
        matrix = np.zeros((len(self.nodes_1), len(self.nodes_2)))
        for i, n in enumerate(self.nodes_1):
            for j, m in enumerate(self.nodes_2):
                t1 = self.g1.nodes[n].get('type')
                t2 = self.g2.nodes[m].get('type')
                
                is_scaffold = (n in locked_scaffold and locked_scaffold[n] == m)
                is_phase1_map = ("PHASE 1" in phase_name and n in mapping and mapping[n] == m)
                is_phase3_map = ("PHASE 3" in phase_name and n in mapping and mapping[n] == m)

                if is_scaffold or is_phase1_map: matrix[i, j] = 4 
                elif is_phase3_map: matrix[i, j] = 5 
                elif n == candidate_n and m == candidate_m: matrix[i, j] = 3 if is_rejection else 2 
                elif t1 == t2 and t1 is not None and n not in locked_scaffold and m not in locked_scaffold.values(): matrix[i, j] = 1 

        self.ax3.matshow(matrix, cmap=self.cmap, vmin=0, vmax=5)
        self.ax3.set_xticks(range(len(self.nodes_2)))
        self.ax3.set_yticks(range(len(self.nodes_1)))
        
        self.ax3.set_xticklabels([f"{m}$_p$ ({self.g2.nodes[m].get('type','')})" for m in self.nodes_2], rotation=45, ha='left', fontsize=10)
        self.ax3.set_yticklabels([f"{n}$_r$ ({self.g1.nodes[n].get('type','')})" for n in self.nodes_1], fontsize=10)
        self.ax3.tick_params(colors='#555555', bottom=False, top=False, left=False, right=False, pad=5)
        for spine in self.ax3.spines.values():
            spine.set_visible(False)
            
        self.ax3.set_xticks(np.arange(-0.5, len(self.nodes_2), 1), minor=True)
        self.ax3.set_yticks(np.arange(-0.5, len(self.nodes_1), 1), minor=True)
        self.ax3.grid(which="minor", color="white", linestyle='-', linewidth=2)

        # --- GRAPHS COLORING ---
        colors1, colors2 = [], []
        if phase_name == "PIPELINE COMPLETE":
            full_mapping = {**locked_scaffold, **mapping}
            color_dict_1, color_dict_2 = {}, {}
            sorted_reactants = sorted([n for n in self.nodes_1 if n in full_mapping])
            rainbow_cmap = plt.get_cmap('rainbow')

            for idx, n in enumerate(sorted_reactants):
                c = rainbow_cmap(idx / max(1, len(sorted_reactants) - 1))
                color_dict_1[n] = c
                if n in full_mapping:
                    color_dict_2[full_mapping[n]] = c

            colors1 = [color_dict_1.get(n, '#f0f3f5') for n in self.nodes_1]
            colors2 = [color_dict_2.get(m, '#f0f3f5') for m in self.nodes_2]
        else:
            for n in self.nodes_1:
                if n == candidate_n: colors1.append('#ff7675' if is_rejection else '#ffeaa7')
                elif n in locked_scaffold: colors1.append('#00b894') 
                elif n in mapping: colors1.append('#00b894' if "PHASE 1" in phase_name else '#00cec9')
                else: colors1.append('#f4f6f9')
            for m in self.nodes_2:
                if m == candidate_m: colors2.append('#ff7675' if is_rejection else '#ffeaa7')
                elif m in locked_scaffold.values(): colors2.append('#00b894')
                elif m in mapping.values(): colors2.append('#00b894' if "PHASE 1" in phase_name else '#00cec9')
                else: colors2.append('#f4f6f9')

        # --- DRAW NODES & EDGES ---
        nx.draw_networkx_nodes(self.g1, self.pos1, ax=self.ax1, node_color=colors1, node_size=800, edgecolors='#cbd5e1', linewidths=2)
        nx.draw_networkx_nodes(self.g2, self.pos2, ax=self.ax2, node_color=colors2, node_size=800, edgecolors='#cbd5e1', linewidths=2)

        full_mapping = {**locked_scaffold, **mapping}
        self._draw_chemical_bonds(self.g1, self.pos1, self.ax1, highlight_edges=broken_bonds, highlight_color='#ff7675', full_mapping=full_mapping, target_G=self.g2, is_g1=True)
        self._draw_chemical_bonds(self.g2, self.pos2, self.ax2, highlight_edges=formed_bonds, highlight_color='#00b894', full_mapping=full_mapping, target_G=self.g1, is_g1=False)

        # --- DRAW LABELS ---
        type_labels_1 = {n: self.g1.nodes[n].get('type', '') for n in self.nodes_1}
        type_labels_2 = {m: self.g2.nodes[m].get('type', '') for m in self.nodes_2}
        nx.draw_networkx_labels(self.g1, self.pos1, labels=type_labels_1, ax=self.ax1, font_size=12, font_weight='bold', font_color='black')
        nx.draw_networkx_labels(self.g2, self.pos2, labels=type_labels_2, ax=self.ax2, font_size=12, font_weight='bold', font_color='black')

        id_labels_1 = {n: f"{n}" for n in self.nodes_1}
        id_labels_2 = {m: f"{m}" for m in self.nodes_2}
        nx.draw_networkx_labels(self.g1, self.pos1_id, labels=id_labels_1, ax=self.ax1, font_size=10, font_weight='bold', font_color='black')
        nx.draw_networkx_labels(self.g2, self.pos2_id, labels=id_labels_2, ax=self.ax2, font_size=10, font_weight='bold', font_color='black')
        
        self.ax1.set_aspect('equal')
        self.ax2.set_aspect('equal')
        self.ax1.margins(0.2)
        self.ax2.margins(0.2)
        
        return self._fig_to_pil()

# ==========================================
# ALGORITHMS 
# ==========================================
class VF2_MCS_Matcher:
    def __init__(self, g1, g2, viz):
        self.g1 = g1
        self.g2 = g2
        self.viz = viz
        self.core_1 = {}
        self.core_2 = {}
        self.best_mapping = {}
        self.max_size = 0

    def find_mcs(self):
        self._match(list(self.g1.nodes()))
        return self.best_mapping

    def _match(self, unmapped_1):
        current_size = len(self.core_1)
        if current_size > self.max_size:
            self.max_size = current_size
            self.best_mapping = self.core_1.copy()

        if current_size + len(unmapped_1) <= self.max_size or not unmapped_1: return

        n = unmapped_1[0]
        remaining_1 = unmapped_1[1:]
        
        # 👇 CHANGE THESE LINES: Get the atom type of 'n' first
        t1 = self.g1.nodes[n].get('type')
        
        # 👇 PRE-FILTER: Only select product nodes (m) that have the EXACT SAME atom type
        unmapped_2 = [m for m in self.g2.nodes() if m not in self.core_2 and self.g2.nodes[m].get('type') == t1]
        
        for m in unmapped_2:
            global_str = f"MCS Best Size so far: {self.max_size}\n{format_mapping(self.best_mapping)}"
            self.viz.record_state("PHASE 1: MCS Scaffold", f"Evaluating {n}$_r$ ⟷ {m}$_p$", self.core_1, global_str, n, m)
            is_valid, reason = self._is_feasible(n, m)
            
            if is_valid:
                self.viz.record_state("PHASE 1: MCS Scaffold", f"Feasible! Temporarily mapping {n}$_r$ ⟷ {m}$_p$", self.core_1, global_str, n, m)
                self.core_1[n] = m
                self.core_2[m] = n
                self._match(remaining_1)
                del self.core_1[n]
                del self.core_2[m]
            else:
                self.viz.record_state("PHASE 1: MCS Scaffold", f"Rejected: {reason}", self.core_1, global_str, n, m)

        self._match(remaining_1)

    def _is_feasible(self, n, m):
        t1 = self.g1.nodes[n].get('type')
        t2 = self.g2.nodes[m].get('type')
        if t1 != t2: return False, f"Semantic Fail: '{t1}' != '{t2}'"
        
        discrepancy_score = 0
        checked_pairs = set()
        
        # --- 1. CORE TOPOLOGY CHECK ---
        # Check G1 edges against G2 for ALREADY MAPPED neighbors
        for n_neighbor in self.g1.neighbors(n):
            if n_neighbor in self.core_1:
                m_neighbor = self.core_1[n_neighbor]
                checked_pairs.add((n_neighbor, m_neighbor))
                
                if not self.g2.has_edge(m, m_neighbor):
                    discrepancy_score += self.g1[n][n_neighbor]['order']
                else:
                    o1 = self.g1[n][n_neighbor]['order']
                    o2 = self.g2[m][m_neighbor]['order']
                    discrepancy_score += abs(o1 - o2)
                    
        # Check G2 edges against G1 for ALREADY MAPPED neighbors
        for m_neighbor in self.g2.neighbors(m):
            if m_neighbor in self.core_2:
                n_neighbor = self.core_2[m_neighbor]
                if (n_neighbor, m_neighbor) not in checked_pairs:
                    discrepancy_score += self.g2[m][m_neighbor]['order']

        # --- 2. UNMAPPED NEIGHBORHOOD CHECK ---
        # Tally the bond orders per atom type for UNMAPPED neighbors
        unmapped_env_1 = {}
        for n_neighbor in self.g1.neighbors(n):
            if n_neighbor not in self.core_1:
                ntype = self.g1.nodes[n_neighbor].get('type')
                order = self.g1[n][n_neighbor]['order']
                unmapped_env_1[ntype] = unmapped_env_1.get(ntype, 0) + order
                
        unmapped_env_2 = {}
        for m_neighbor in self.g2.neighbors(m):
            if m_neighbor not in self.core_2:
                mtype = self.g2.nodes[m_neighbor].get('type')
                order = self.g2[m][m_neighbor]['order']
                unmapped_env_2[mtype] = unmapped_env_2.get(mtype, 0) + order
                
        # Compare the unmapped environments and add differences to the score
        all_types = set(unmapped_env_1.keys()).union(set(unmapped_env_2.keys()))
        for t in all_types:
            o1 = unmapped_env_1.get(t, 0)
            o2 = unmapped_env_2.get(t, 0)
            discrepancy_score += abs(o1 - o2)
                    
        # --- 3. FINAL VERDICT ---
        if discrepancy_score > 1.0:
            return False, f"Structural Fail: Total discrepancy ({discrepancy_score}) exceeds limit of 1.0"
            
        return True, "Valid"

class Phase3_ReactionCenterMapper:
    def __init__(self, full_g1, full_g2, scaffold_mapping, unmapped_1, unmapped_2, viz):
        self.g1 = full_g1
        self.g2 = full_g2
        self.scaffold = scaffold_mapping
        self.unmapped_1 = unmapped_1
        self.unmapped_2 = unmapped_2
        self.viz = viz
        
        self.best_mapping = {}
        self.min_cost = float('inf')
        self.best_broken = []
        self.best_formed = []

    def solve(self):
        initial_str = f"Isolated Reaction Center:\nReactants: {[f'{x}$_r$' for x in self.unmapped_1]}\nProducts: {[f'{x}$_p$' for x in self.unmapped_2]}"
        self.viz.record_state("PHASE 2: Start", "Identifying unmapped atoms...", {}, initial_str, locked_scaffold=self.scaffold)
        
        self._search({}, self.unmapped_1, self.unmapped_2)
        return self.best_mapping, self.min_cost, self.best_broken, self.best_formed

    def _search(self, current_rc_mapping, remaining_1, unmapped_2):
        c_str = "None" if self.min_cost == float('inf') else str(self.min_cost)
        global_str = f"Phase 2 Lowest Cost so far: {c_str}\n{format_mapping(self.best_mapping)}"
        
        if remaining_1:
            n = remaining_1[0]
            next_remaining = remaining_1[1:]
            
            t1 = self.g1.nodes[n].get('type')
            valid_candidates = [m for m in unmapped_2 if self.g2.nodes[m].get('type') == t1]
            
            for m in valid_candidates:
                self.viz.record_state("PHASE 2: Chem Distance", f"Evaluating & Mapping {n}$_r$ ⟷ {m}$_p$", current_rc_mapping, global_str, n, m, self.scaffold)
                
                current_rc_mapping[n] = m
                next_unmapped_2 = [x for x in unmapped_2 if x != m]
                self._search(current_rc_mapping, next_remaining, next_unmapped_2)
                del current_rc_mapping[n]
                    
            self._search(current_rc_mapping, next_remaining, unmapped_2)
            
        else:
            cost, broken, formed, penalty = self._calculate_chemical_distance(current_rc_mapping)
            
            if cost < self.min_cost:
                self.min_cost = cost
                self.best_mapping = current_rc_mapping.copy()
                self.best_broken = broken
                self.best_formed = formed
                global_str = f"Phase 2 Lowest Cost so far: {self.min_cost}\n{format_mapping(self.best_mapping)}"
                
            bond_cost = cost - penalty
            status_text = f"Cost Math:\n{bond_cost} (Bond Changes) + {penalty} (Unmapped Penalty) = {cost}"
            self.viz.record_state("PHASE 2: Evaluation", status_text, current_rc_mapping, global_str, locked_scaffold=self.scaffold, broken_bonds=broken, formed_bonds=formed)

    def _calculate_chemical_distance(self, rc_mapping):
        full_mapping = {**self.scaffold, **rc_mapping}
        broken, formed = [], []
        cost = 0
        
        unmapped_penalty = (len(self.g1.nodes()) - len(full_mapping)) * 10
        unmapped_penalty += (len(self.g2.nodes()) - len(full_mapping)) * 10
        cost += unmapped_penalty
        
        for u, v, data in self.g1.edges(data=True):
            o1 = data['order']
            if u in full_mapping and v in full_mapping:
                m_u, m_v = full_mapping[u], full_mapping[v]
                if self.g2.has_edge(m_u, m_v):
                    o2 = self.g2[m_u][m_v]['order']
                    if o1 > o2: 
                        broken.append((u, v))
                        cost += (o1 - o2)
                    elif o2 > o1: 
                        formed.append((m_u, m_v)) 
                        cost += (o2 - o1)
                else: 
                    broken.append((u, v))
                    cost += o1
            else: 
                broken.append((u, v))
                cost += o1
                
        inv_mapping = {v: k for k, v in full_mapping.items()}
        for u, v, data in self.g2.edges(data=True):
            o2 = data['order']
            if u in inv_mapping and v in inv_mapping:
                m_u, m_v = inv_mapping[u], inv_mapping[v]
                if not self.g1.has_edge(m_u, m_v):
                    formed.append((u, v))
                    cost += o2
            else:
                formed.append((u, v))
                cost += o2
                
        return cost, broken, formed, unmapped_penalty

class HybridReactionMapper:
    def __init__(self, reactants_graph, products_graph, pos1, pos2, viz=None):
        self.g1 = reactants_graph
        self.g2 = products_graph
        self.viz = viz if viz else ReactionVisualizer(self.g1, self.g2, pos1, pos2)
        
    def map_reaction(self):
        self.viz.record_state("PHASE 1: Finding Maximum Common Substructure (MCS) Scaffold", "Start searching by moving the slider...", {}, "MCS Best Size so far: 0")
        mcs_matcher = VF2_MCS_Matcher(self.g1, self.g2, self.viz)
        scaffold_mapping = mcs_matcher.find_mcs()
        
        str_scaffold = f"MCS Scaffold locked in.\nSize: {len(scaffold_mapping)}\n{format_mapping(scaffold_mapping)}"
        self.viz.record_state("PHASE 1 COMPLETE", f"Max Size: {len(scaffold_mapping)}", scaffold_mapping, str_scaffold)
        
        unmapped_reactants = [n for n in self.g1.nodes() if n not in scaffold_mapping.keys()]
        unmapped_products = [m for m in self.g2.nodes() if m not in scaffold_mapping.values()]
        
        distance_matcher = Phase3_ReactionCenterMapper(self.g1, self.g2, scaffold_mapping, unmapped_reactants, unmapped_products, self.viz)
        rxn_center_mapping, rxn_cost, rxn_broken, rxn_formed = distance_matcher.solve()
        
        rxn_broken_str = ", ".join([f"({u}$_r$, {v}$_r$)" for u, v in rxn_broken]) if rxn_broken else "None"
        rxn_formed_str = ", ".join([f"({u}$_p$, {v}$_p$)" for u, v in rxn_formed]) if rxn_formed else "None"
        final_mapping = {**scaffold_mapping, **rxn_center_mapping}

        str_final = f"PIPELINE COMPLETE.\nFinal Mapping:\n{format_mapping(final_mapping)}\nTotal Bond Cost: {rxn_cost}\nBonds Broken: {rxn_broken_str}\nBonds Formed: {rxn_formed_str}"
        
        
        self.viz.record_state("PIPELINE COMPLETE", "", rxn_center_mapping, str_final, locked_scaffold=scaffold_mapping, broken_bonds=rxn_broken, formed_bonds=rxn_formed)
        
        return self.viz

# ==========================================
# STREAMLIT INTERFACE
# ==========================================

# Initialize Session States
if 'viz' not in st.session_state:
    st.session_state.viz = None
if 'step' not in st.session_state:
    st.session_state.step = -1
if 'status' not in st.session_state:
    st.session_state.status = "Ready to calculate."
if 'slider_step' not in st.session_state:
    st.session_state.slider_step = 0

st.title("Algorithmic Atom-to-Atom Mapper ⚛️")

# Top controls
col1, col2, col3, col4 = st.columns([3, 3, 1, 1])

r_input = col1.text_input("Reactants SMILES", value="CC(Cl)(C)CCC", key="r_input")
p_input = col2.text_input("Products SMILES", value="CC(C)=CCC", key="p_input")

# Handlers for Buttons
def run_preview():
    if st.session_state.viz is not None:
        plt.close(st.session_state.viz.fig)
    try:
        G1, pos1 = smiles_to_graph(st.session_state.r_input)
        G2, pos2 = smiles_to_graph(st.session_state.p_input)
        st.session_state.viz = ReactionVisualizer(G1, G2, pos1, pos2)
        st.session_state.step = -1  # Indicates preview mode
        st.session_state.status = "Preview generated. Click 'Calculate Mapping' to run the algorithm."
    except Exception as e:
        st.error(f"Error parsing SMILES: {e}")

def run_calculation():
    if st.session_state.viz is not None:
        plt.close(st.session_state.viz.fig)
    try:
        G1, pos1 = smiles_to_graph(st.session_state.r_input)
        G2, pos2 = smiles_to_graph(st.session_state.p_input)
        viz = ReactionVisualizer(G1, G2, pos1, pos2)
        mapper = HybridReactionMapper(viz.g1, viz.g2, viz.pos1, viz.pos2, viz)
        st.session_state.viz = mapper.map_reaction()
        
        st.session_state.step = 0
        st.session_state.slider_step = 0
        st.session_state.status = "Run complete! Use the slider to go through each step or press the buttons to jump between phases."
    except Exception as e:
        st.error(f"Error parsing SMILES: {e}")

def set_milestone(milestone):
    viz = st.session_state.viz
    if not viz or not viz.history: 
        return
        
    if milestone == "MCS":
        st.session_state.slider_step = 0
    elif milestone == "Unmapped":
        idx_p1 = next((i for i, state in enumerate(viz.history) if state['phase_name'] == "PHASE 1 COMPLETE"), 0)
        st.session_state.slider_step = next((i for i, state in enumerate(viz.history) if state['phase_name'] == "PHASE 3: Start"), idx_p1)
    elif milestone == "Final":
        st.session_state.slider_step = len(viz.history) - 1

# Render Buttons
col3.button("🔍 Preview Molecules", on_click=run_preview, use_container_width=True)
col4.button("🗺️ Calculate Mapping", type="primary", on_click=run_calculation, use_container_width=True)

st.markdown(st.session_state.status)

# Display Slider and Navigation if history exists
if st.session_state.viz is not None and st.session_state.step >= 0:
    max_idx = len(st.session_state.viz.history) - 1
        
    mc1, mc2, mc3 = st.columns([1, 1, 1])
    mc1.button("MCS", on_click=set_milestone, args=("MCS",), use_container_width=True)
    mc2.button("Still Unmapped", on_click=set_milestone, args=("Unmapped",), use_container_width=True)
    mc3.button("Final Result", on_click=set_milestone, args=("Final",), use_container_width=True)



# ==========================================
# STREAMLIT INTERFACE
# ==========================================

# Render Display Image
if st.session_state.viz is not None:
    # 1. Fetch the image and the correct text strings based on the current step
    if st.session_state.step == -1: # Preview mode
        combined_text = "### Preview Mode"
        status_md = ""
        img = st.session_state.viz.get_preview_frame()
    else:
        # Pull phase and status directly from the visualizer's history
        state = st.session_state.viz.history[st.session_state.slider_step]
        is_rejection = "Rejected" in state['status'] or "Fail" in state['status']
        
        phase_md = f"### {state['phase_name']}"
        
        if is_rejection:
            status_md = f"**:red[{state['status']}]**"
        else:
            status_md = f"**{state['status']}**"
        combined_text = f"{phase_md}  \n{status_md}"
        img = st.session_state.viz.get_frame(st.session_state.slider_step)
        
    # 2. Render the titles natively in Streamlit (above the image)
    # st.markdown(phase_md)
    # if status_md:
    #     st.markdown(status_md) # No unsafe_allow_html needed!
        
    # 3. Render the image in the centered column
    spacer_left, center_col, spacer_right = st.columns([1, 3, 1])    
    st.markdown(combined_text)   
    if st.session_state.step >= 0:
        st.slider("Algorithm Step", min_value=0, max_value=max_idx, step=1, key="slider_step")
    st.image(img, width=1400)
