# --- inside draw_example(), after we have (pl, pt, pr, pb), cx, cy and 'axis' ---

if axis == "V":
    # Vertical fold: show the arrow sweeping ACROSS the vertical dotted line.
    dotted_line_circles(rd, (cx, pt + 10), (cx, pb - 10), fold_line_color, dot_r=4, gap=16)

    Wp = pr - pl
    Hp = pb - pt
    # Arc box spans BOTH halves horizontally so it must cross the dotted line
    arc_left   = pl + int(0.06 * Wp)
    arc_right  = pr - int(0.06 * Wp)
    # Keep the arc comfortably inside the paper vertically
    arc_top    = pt + int(0.18 * Hp)
    arc_bottom = pt + int(0.78 * Hp)

    # L = right half folds over to the left; R = left half folds over to the right
    if direction == "L":
        start_deg, end_deg = (210, -20)   # sweeps from upper-right toward upper-left (crosses center)
    else:  # direction == "R"
        start_deg, end_deg = (-20, 210)   # sweeps from upper-left toward upper-right (crosses center)

    draw_arc_arrow(rd, (arc_left, arc_top, arc_right, arc_bottom),
                   start_deg, end_deg, outline, width=6, head_len=18, head_w=12)

else:
    # Horizontal fold: arrow sweeps ACROSS the horizontal dotted line.
    dotted_line_circles(rd, (pl + 10, cy), (pr - 10, cy), fold_line_color, dot_r=4, gap=16)

    Wp = pr - pl
    Hp = pb - pt
    # Arc box spans BOTH halves vertically so it crosses the dotted line
    arc_left   = pl + int(0.10 * Wp)
    arc_right  = pr - int(0.10 * Wp)
    arc_top    = pt + int(0.06 * Hp)
    arc_bottom = pb - int(0.06 * Hp)

    # U = bottom half folds up; D = top half folds down
    if direction == "U":
        start_deg, end_deg = (160, 340)   # sweeps upward across the center
    else:  # direction == "D"
        start_deg, end_deg = (-20, 160)   # sweeps downward across the center

    draw_arc_arrow(rd, (arc_left, arc_top, arc_right, arc_bottom),
                   start_deg, end_deg, outline, width=6, head_len=18, head_w=12)
