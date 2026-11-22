import ao_ra as m
p,a,d,b = m.compute_utility_vs_power(18,20)
for i,x in enumerate(p): print(x, round(a[i],2), round(d[i],2), round(b[i],2))