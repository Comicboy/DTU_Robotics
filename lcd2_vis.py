import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.linspace(0, 10, 5000)

# Define modes: real poles (pure exponential) and complex pairs (damped oscillations)
# α = real part, ω = imaginary part (0 for real poles)
modes = [
    {"alpha": -0.5, "omega": 3.0, "A": 1.0},   # complex pair 1
    {"alpha": -1.5, "omega": 6.0, "A": 0.6},   # complex pair 2
    {"alpha": -3.0, "omega": 9.0, "A": 0.3},   # complex pair 3
    {"alpha": -0.8, "omega": 0.0, "A": 0.8},   # real pole 1
    {"alpha": -2.5, "omega": 0.0, "A": 0.4},   # real pole 2
]

responses = []
envelopes = []

for i, m in enumerate(modes):
    # Derived quantities
    omega_n = np.sqrt(m["alpha"]**2 + m["omega"]**2)
    zeta = -m["alpha"] / omega_n if m["omega"] != 0 else 1.0  # ζ=1 for real (non-oscillatory)
    m["omega_n"] = omega_n
    m["zeta"] = zeta

    # Compute response
    if m["omega"] == 0:
        # Real pole → pure exponential
        y = m["A"] * np.exp(m["alpha"] * t)
        env = np.abs(y)
    else:
        # Complex pole pair → damped oscillation
        y = m["A"] * np.exp(m["alpha"] * t) * np.sin(m["omega"] * t)
        env = m["A"] * np.exp(m["alpha"] * t)

    responses.append(y)
    envelopes.append(env)

# Superposition (total response)
y_total = np.sum(responses, axis=0)

# Plot setup
plt.figure(figsize=(12, 8))

# Individual modes with envelopes
for i, m in enumerate(modes):
    color = plt.cm.tab10(i)
    if m["omega"] == 0:
        label = f"Real Mode {i+1}: α={m['alpha']}, pure exponential"
    else:
        label = (f"Complex Mode {i+1}: α={m['alpha']}, ω={m['omega']}, "
                 f"ωₙ={m['omega_n']:.2f}, ζ={m['zeta']:.2f}")

    plt.plot(t, responses[i], '--', color=color, label=label)
    plt.plot(t, envelopes[i], ':', color=color, alpha=0.6)
    if m["omega"] != 0:
        plt.plot(t, -envelopes[i], ':', color=color, alpha=0.6)

# Total response
plt.plot(t, y_total, 'k', linewidth=2, label="Total response (superposition)")

# Formatting
plt.title("Superposition of Real and Complex Modes in an LTI System", fontsize=14)
plt.xlabel("Time [s]")
plt.ylabel("Response amplitude")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
