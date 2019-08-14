from rdp_accountant import compute_rdp, compute_rdp_sample_without_replacement
import numpy as np


#  A simple test script to demonstrate the calculated RDP

q= 0.01
noise_multiplier = 2
steps = 10
orders = np.linspace(2,100,99)

results1 = compute_rdp(q, noise_multiplier, steps,orders)

results2 = compute_rdp_sample_without_replacement(q, noise_multiplier, steps, orders)


import matplotlib.pyplot as plt

plt.loglog(orders, results1,label='Poisson sampling')
plt.loglog(orders, results2,label='Sample without replacement')
plt.legend()
plt.show()