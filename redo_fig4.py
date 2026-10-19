import ao_ra
from ao_ra import apply_ieee_style, plot_fig5_oru_load

apply_ieee_style()
plot_fig5_oru_load(D_fixed=27, seed=42)
print("done")