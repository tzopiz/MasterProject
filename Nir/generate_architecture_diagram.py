#!/usr/bin/env python3
"""
Инженерная схема архитектуры системы (Hub & Spoke layout)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

# Настройка стиля
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

# Создаем фигуру
fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Цветовая палитра (Professional UI)
colors = {
    'client': '#2980B9',   # Blue
    'backend': '#8E44AD',  # Purple
    'ml': '#E67E22',       # Orange
    'db': '#27AE60',       # Green
    'text': '#2C3E50',
    'bg_card': '#FFFFFF',
    'line': '#7F8C8D'
}

def draw_service_card(ax, x, y, w, h, title, icon_char, color_key):
    main_color = colors[color_key]
    
    # Shadow (rounded)
    shadow = FancyBboxPatch((x+0.05, y-0.05), w, h, 
                            boxstyle="round,pad=0.0,rounding_size=0.2", 
                            facecolor='#BDC3C7', alpha=0.5, zorder=1)
    ax.add_patch(shadow)
    
    # Main Box Border (rounded)
    box = FancyBboxPatch((x, y), w, h, 
                         boxstyle="round,pad=0.0,rounding_size=0.2", 
                         facecolor=colors['bg_card'], edgecolor=main_color, linewidth=2, zorder=2)
    ax.add_patch(box)
    
    # Colored Header (rounded top only - simulated by clipping or overlay)
    # Simpler approach: Draw rounded rect for header, then fill bottom part to square it off 
    # where it meets the body, BUT since body is rounded, we need the header to be rounded at top 
    # and straight at bottom.
    
    header_h = h * 0.35
    
    # 1. Full rounded header background
    header_path = FancyBboxPatch((x, y + h - header_h), w, header_h, 
                                 boxstyle="round,pad=0.0,rounding_size=0.2", 
                                 facecolor=main_color, edgecolor='none', zorder=3)
    ax.add_patch(header_path)
    
    # 2. Rectangle to "un-round" the bottom corners of the header 
    # (so it connects seamlessly with the white body)
    # Actually, we want the connection line to be straight. 
    # Drawing a straight rect at the bottom of the header area works.
    rect_straighten = mpatches.Rectangle((x, y + h - header_h), w, header_h/2, 
                                         facecolor=main_color, zorder=3)
    ax.add_patch(rect_straighten)
    
    # 3. Re-draw the border of the main box on top to ensure clean edges? 
    # No, box border is already drawn.
    
    # Title in Header
    ax.text(x + w/2, y + h - header_h/2, title, 
            ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white', zorder=4)
    
    # Icon/Content placeholder in body (Text instead of Emoji)
    ax.text(x + w/2, y + (h - header_h)/2, icon_char, 
            ha='center', va='center', 
            fontsize=16, fontweight='bold', color=main_color, alpha=0.3, zorder=4)

def draw_ortho_arrow(ax, x1, y1, x2, y2, label, label_pos=0.5):
    """Рисует ортогональную или прямую стрелку с подписью"""
    
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>,head_width=5,head_length=10',
        color=colors['line'],
        linewidth=2,
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x = x1 + (x2 - x1) * label_pos
        mid_y = y1 + (y2 - y1) * label_pos
        
        bbox = dict(boxstyle="round,pad=0.3", fc="white", ec=colors['line'], alpha=1.0)
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=9, color=colors['text'], bbox=bbox, zorder=5)

# Layout Configuration
card_w = 2.5
card_h = 1.8
center_y = 4.0

# Coordinates
# Client (Left)
client_x = 1.0
client_y = center_y

# Backend (Center)
backend_x = 5.0
backend_y = center_y

# ML (Right)
ml_x = 9.0
ml_y = center_y

# DB (Bottom Center)
db_x = 5.0
db_y = 1.0

# --- Draw Nodes ---

# 1. Client App
draw_service_card(ax, client_x, client_y, card_w, card_h, "iOS App", "CLIENT", 'client')

# 2. Backend
draw_service_card(ax, backend_x, backend_y, card_w, card_h, "Backend", "SERVER", 'backend')

# 3. ML Service
draw_service_card(ax, ml_x, ml_y, card_w, card_h, "ML Service", "AI / GPU", 'ml')

# 4. Database
draw_service_card(ax, db_x, db_y, card_w, card_h, "Database", "SQL", 'db')


# --- Draw Edges ---

# Client <-> Backend
# Two arrows for Request/Response or one double-headed? 
# User asked for descriptions. Let's do distinct lines.

# Request (Top)
draw_ortho_arrow(ax, client_x + card_w, client_y + card_h*0.6, 
                 backend_x, backend_y + card_h*0.6, "DICOM")

# Response (Bottom)
draw_ortho_arrow(ax, backend_x, backend_y + card_h*0.4, 
                 client_x + card_w, client_y + card_h*0.4, "Результат")


# Backend <-> ML
# Request (Top)
draw_ortho_arrow(ax, backend_x + card_w, backend_y + card_h*0.6, 
                 ml_x, ml_y + card_h*0.6, "Задача")

# Response (Bottom)
draw_ortho_arrow(ax, ml_x, ml_y + card_h*0.4, 
                 backend_x + card_w, backend_y + card_h*0.4, "JSON")


# Backend <-> DB
# Down
draw_ortho_arrow(ax, backend_x + card_w*0.3, backend_y, 
                 db_x + card_w*0.3, db_y + card_h, "Save")

# Up
draw_ortho_arrow(ax, db_x + card_w*0.7, db_y + card_h, 
                 backend_x + card_w*0.7, backend_y, "Load")


plt.tight_layout()
plt.savefig('/Users/tzopiz/Developer/MasterProject/Nir/architecture_diagram.svg', 
            format='svg', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('/Users/tzopiz/Developer/MasterProject/Nir/architecture_diagram.png', 
            format='png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Инженерная схема (Hub & Spoke) создана!")
