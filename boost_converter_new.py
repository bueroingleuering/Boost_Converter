"""@author: robin leuering"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
"""Parameters"""
f_Switch = 50.0e3  # 50 kHz
Duty = 0.5  # 50 %
U_0 = 100.0  # 100 V
I_max = 10.0  # 10 A
L_0 = 250.0e-6  # 0.25 mH
R_L = 10.0e-3  # 0.01 Ω
R_DS = 20.0e-3  # 0.02 Ω
C_0 = 25.0e-6  # 25 µF
R_Load =  U_0 / ((1 - Duty) * I_max)  # 20 Ω
U_C_Init = U_0
"""Simulation Parameter"""
T_Switch = 1 / f_Switch  # 0.02 ms
T_Sim = 500 * T_Switch  # 10 ms
T_Step = T_Switch / 100.0  # 0.0002 ms
"""switch mode"""
t = np.arange(start=0, stop=T_Sim, step=T_Step)
gate_pulse = signal = square(2 * np.pi * f_Switch * t, duty=Duty)
i = U_C_Init / R_Load * np.ones_like(t)
"""mean method"""
i_m = U_C_Init / R_Load * np.ones_like(t)
U_C = U_C_Init * np.ones_like(t)
U_Cm = U_C_Init * np.ones_like(t)
"""state space"""
A = np.array([[-(R_L + Duty * R_DS) / L_0, -(1 - Duty) / L_0], [(1 - Duty) / C_0, -1 / (R_Load * C_0)]])
B, C, D = np.array([[1 / L_0], [0.0]]), np.array([[0.0, 1.0]]), np.array([0.0])
x = np.array([U_C_Init / R_Load, U_C_Init])
X = np.zeros((2, len(t)))
X[:, 0] = x
u = U_0 * np.ones_like(t)
"""Simulation"""
for index in range(1 ,len(t)):
    """switch mode"""
    if gate_pulse[index] > 0:
        didt = (U_0 - (R_DS + R_L) * i[index - 1]) / L_0
        dU_Cdt = (-U_C[index - 1] / R_Load ) / C_0
    else:
        didt = (U_0 - R_L * i[index - 1] - U_C[index - 1]) / L_0
        dU_Cdt = (i[index - 1] - U_C[index - 1] / R_Load ) / C_0
    i[index] = i[index - 1] + didt * T_Step
    U_C[index] = U_C[index - 1] + dU_Cdt * T_Step
    """mean method"""
    di_mdt = (U_0 - (Duty * R_DS + R_L) * i_m[index - 1] - (1 - Duty) * U_Cm[index - 1]) / L_0
    dU_Cmdt = ((1 - Duty) * i_m[index - 1] - U_Cm[index - 1] / R_Load) / C_0
    i_m[index] = i_m[index - 1] + di_mdt * T_Step
    U_Cm[index] = U_Cm[index - 1] + dU_Cmdt * T_Step
    """state space"""
    x_dot = A @ x + B.flatten().dot(u[index - 1])
    x = x + x_dot * T_Step
    X[:, index] = x
i_s = X[0, :]
U_Cs = X[1, :]
"""Plots"""
fig_1 = plt.figure(num=1, figsize=(10, 6))
plt.plot(t, i, label="current switched", color='#284b64', linewidth=3)
plt.plot(t, i_s, label="current state space", color='#3C6E71', linewidth=3)
plt.plot(t, i_m, label="current mean", color='#893636', linewidth=3, linestyle='--')
plt.xlabel('Time [s]', fontsize=16), plt.ylabel('Current [A]', fontsize=16), plt.legend(fontsize=16), plt.grid()
plt.savefig('current.png', format="png", dpi=300)

fig_2 = plt.figure(num=2, figsize=(10, 6))
plt.plot(t, U_C, label="voltage switched", color='#284b64', linewidth=3)
plt.plot(t, U_Cs, label="voltage state space", color='#3C6E71', linewidth=3)
plt.plot(t, U_Cm, label="voltage mean", color='#893636', linewidth=3, linestyle='--')
plt.xlabel('Time [s]', fontsize=16), plt.ylabel('Voltage [V]', fontsize=16), plt.legend(fontsize=16), plt.grid()
plt.savefig('voltage.png', format="png", dpi=300)