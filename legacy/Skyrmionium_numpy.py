"""
Skyrmionium 量子输运模拟 (基于 numpy/scipy)
计算电子通过单个 Skyrmionium (Q=0) 自旋纹理的 Landauer 透射谱 T(E)
使用 递归格林函数 (RGF) + Lopez-Sancho 算法
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. Skyrmionium 磁化分布 (Q = 0)
# ============================================================
def get_skyrmionium_m(x, y, x0, y0, R):
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)

    # Theta: 0 -> 2pi, passing pi at r=R/2 (Skyrmionium core feature)
    theta = np.where(r <= R, np.pi * (1.0 - np.cos(np.pi * r / R)), 0.0)

    mx = np.sin(theta) * np.cos(phi)
    my = np.sin(theta) * np.sin(phi)
    mz = np.cos(theta)
    return mx, my, mz


def skyrmion_number(L, W, R):
    """Lattice-solid-angle evaluation of the skyrmion number Q."""
    x, y = np.meshgrid(np.arange(L), np.arange(W), indexing='ij')
    x0, y0 = (L - 1) / 2, (W - 1) / 2
    mx, my, mz = get_skyrmionium_m(x, y, x0, y0, R)
    m = np.stack((mx, my, mz), axis=-1)

    def solid_angle(a, b, c):
        numerator = np.sum(a * np.cross(b, c), axis=-1)
        denominator = (1 + np.sum(a * b, axis=-1) +
                       np.sum(b * c, axis=-1) + np.sum(c * a, axis=-1))
        return 2 * np.arctan2(numerator, denominator)

    m00, m10 = m[:-1, :-1], m[1:, :-1]
    m11, m01 = m[1:, 1:], m[:-1, 1:]
    omega = solid_angle(m00, m10, m11) + solid_angle(m00, m11, m01)
    return np.sum(omega) / (4 * np.pi)


def clean_lead_modes(energies, W, J, t):
    """Ballistic two-terminal transmission of the identical uniform leads.

    The strip has open boundaries in y.  This is the correct reference for
    the textured device: a clean, perfectly matched sample has T(E)=N(E).
    """
    n = np.arange(1, W + 1)
    transverse = -2 * t * np.cos(n * np.pi / (W + 1))
    subband_edges = np.concatenate((transverse - J, transverse + J))
    return np.array([np.count_nonzero(np.abs(E - subband_edges) < 2 * t)
                     for E in energies])

# ============================================================
# 2. 泡利矩阵与基本常量
# ============================================================
s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# ============================================================
# 3. Lopez-Sancho (Sancho-Rubio) 迭代算法求解电极 Surface GF
# ============================================================
def lead_surface_gf_sancho_rubio(E, h_slice, V, eta=1e-7, max_iter=100, tol=1e-10):
    """
    使用 Lopez-Sancho 快速收敛算法计算半无限电极的 Surface Green's Function
    """
    E_c = (E + 1j * eta) * np.eye(h_slice.shape[0], dtype=complex)
    e_i = h_slice.copy()
    e_s = h_slice.copy()
    alpha = V.copy()
    beta = V.conj().T

    for _ in range(max_iter):
        g_i = np.linalg.inv(E_c - e_i)
        a_gi = alpha @ g_i
        b_gi = beta @ g_i
        
        e_s += a_gi @ beta
        e_i += a_gi @ beta + b_gi @ alpha
        
        alpha = a_gi @ alpha
        beta = b_gi @ beta
        
        if np.max(np.abs(alpha)) < tol:
            break
            
    g_surface = np.linalg.inv(E_c - e_s)
    return g_surface

# ============================================================
# 4. 递归格林函数 (RGF) 计算透射率 T(E)
# ============================================================
def compute_transmission(L, W, J, t, R, energies, eta=1e-7):
    dim_slice = 2 * W  # 每个切片 (x列) 的 Hilbert 空间维度
    x0, y0 = (L - 1) / 2, (W - 1) / 2

    # 构建散射区每个切片 (x=0..L-1) 的内部 Hamiltonians
    H_slices = []
    for x in range(L):
        h_sl = np.zeros((dim_slice, dim_slice), dtype=complex)
        for y in range(W):
            mx, my, mz = get_skyrmionium_m(x, y, x0, y0, R)
            h_2x2 = -J * (mx * sx + my * sy + mz * sz)
            # y 方向最近邻跳跃
            if y > 0:
                h_sl[2*y:2*y+2, 2*(y-1):2*(y-1)+2] = -t * s0
                h_sl[2*(y-1):2*(y-1)+2, 2*y:2*y+2] = -t * s0
            h_sl[2*y:2*y+2, 2*y:2*y+2] = h_2x2
        H_slices.append(h_sl)

    # 切片间跳跃矩阵 (x方向): V = -t * I
    V_inter = -t * np.eye(dim_slice, dtype=complex)

    # 电极的切片 Hamiltonian (均匀铁磁背景 m || z)
    h_lead_slice = np.zeros((dim_slice, dim_slice), dtype=complex)
    for y in range(W):
        h_2x2 = -J * sz
        if y > 0:
            h_lead_slice[2*y:2*y+2, 2*(y-1):2*(y-1)+2] = -t * s0
            h_lead_slice[2*(y-1):2*(y-1)+2, 2*y:2*y+2] = -t * s0
        h_lead_slice[2*y:2*y+2, 2*y:2*y+2] = h_2x2

    transmission = []

    for E in energies:
        # 1. 计算半无限电极的 Surface GF 和 Self-Energies
        g_L_surf = lead_surface_gf_sancho_rubio(E, h_lead_slice, V_inter, eta=eta)
        g_R_surf = lead_surface_gf_sancho_rubio(E, h_lead_slice, V_inter, eta=eta)

        Sigma_L = V_inter.conj().T @ g_L_surf @ V_inter
        Sigma_R = V_inter @ g_R_surf @ V_inter.conj().T

        Gamma_L = 1j * (Sigma_L - Sigma_L.conj().T)
        Gamma_R = 1j * (Sigma_R - Sigma_R.conj().T)

        E_c = E + 1j * eta

        # 2. 递归格林函数 (从左向右构建左连通 Surface GF)
        g_L_list = []
        # 第 0 片包含左电极自能 Sigma_L
        g_curr = np.linalg.inv(E_c * np.eye(dim_slice) - H_slices[0] - Sigma_L)
        g_L_list.append(g_curr)

        for x in range(1, L - 1):
            g_curr = np.linalg.inv(E_c * np.eye(dim_slice) - H_slices[x] - V_inter.conj().T @ g_curr @ V_inter)
            g_L_list.append(g_curr)

        # 3. 最后一个切片 (x = L-1) 正确嵌入右电极自能 Sigma_R
        G_LL = np.linalg.inv(E_c * np.eye(dim_slice) - H_slices[-1] - Sigma_R - V_inter.conj().T @ g_L_list[-1] @ V_inter)

        # 4. 端到端传播子 G_{0, L-1}
        P = g_L_list[0]
        for x in range(1, L - 1):
            P = P @ V_inter @ g_L_list[x]
        G_0_L1 = P @ V_inter @ G_LL

        # 5. Fisher-Lee 公式计算透射系数
        T = np.real(np.trace(Gamma_L @ G_0_L1 @ Gamma_R @ G_0_L1.conj().T))
        transmission.append(T)

    return np.array(transmission)

# ============================================================
# 5. 主程序
# ============================================================
if __name__ == "__main__":
    # 系统参数修正：将宽度 W 放宽至 30，确保电子有走廊绕行
    L, W = 60, 30        # 长 60, 宽 30 格点
    J = 1.5               # Hund 耦合强度
    t_hop = 1.0           # 最近邻跳跃
    R = 8                 # Skyrmionium 半径

    print(f"系统尺寸: {L}×{W} 格点")
    print(f"希尔伯特空间维度: {2*L*W}")
    print(f"Hund耦合 J={J}, 跃迁 t={t_hop}, Skyrmionium半径 R={R}")
    print("正在计算透射谱...")

    # 能量扫描范围
    # 使用奇数个能量点，确保严格包含 E=0。
    energies = np.linspace(-3.0, 3.0, 121)
    transmission = compute_transmission(L, W, J, t_hop, R, energies)
    clean_transmission = clean_lead_modes(energies, W, J, t_hop)
    Q = skyrmion_number(L, W, R)

    print("计算完成！")

    # ========== 画图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左图: Skyrmionium 自旋纹理 ---
    ax1 = axes[0]
    grid_size = 60
    xs = np.linspace(0, L-1, grid_size)
    ys = np.linspace(0, W-1, grid_size)
    X, Y = np.meshgrid(xs, ys)
    Mx = np.zeros_like(X)
    My = np.zeros_like(X)
    Mz = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            mx, my, mz = get_skyrmionium_m(
                X[i, j], Y[i, j], (L - 1) / 2, (W - 1) / 2, R
            )
            Mx[i,j] = mx
            My[i,j] = my
            Mz[i,j] = mz

    skip = 4
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               Mx[::skip, ::skip], My[::skip, ::skip],
               Mz[::skip, ::skip], cmap='RdYlBu', clim=[-1, 1],
               scale=30, width=0.003, pivot='mid')

    im = ax1.pcolormesh(X, Y, Mz, cmap='RdYlBu', vmin=-1, vmax=1, alpha=0.4, shading='auto')
    plt.colorbar(im, ax=ax1, label='$m_z$')

    circle = plt.Circle(((L - 1) / 2, (W - 1) / 2), R,
                        fill=False, color='black', linestyle='--', linewidth=2)
    ax1.add_patch(circle)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'Skyrmionium Spin Texture (Q=0)\nθ: 0→π→2π,  R={R}, W={W}', fontsize=13)
    ax1.set_aspect('equal')

    # --- 右图: 透射谱 T(E) ---
    ax2 = axes[1]
    ax2.step(energies, clean_transmission, where='mid', color='0.35',
             ls='--', lw=1.4, label='Matched clean strip $N(E)$')
    ax2.plot(energies, transmission, 'b-', lw=2.5,
             label=r'Skyrmionium ($Q\approx 0$)')

    ax2.fill_between(energies, 0, transmission, alpha=0.15, color='blue')
    ax2.set_xlabel('Energy $E / t$', fontsize=13)
    ax2.set_ylabel('Dimensionless transmission $T(E)$', fontsize=13)
    ax2.set_title('Quantum Transport Spectrum\nof a Single Skyrmionium', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / 'Skyrmionium_transmission.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n====== Transmission Spectrum Features ======")
    i0 = np.argmin(np.abs(energies))
    print(f"Lattice skyrmion number: Q = {Q:.3e}")
    print(f"Max T: {np.max(transmission):.3f}")
    print(f"T(E=0): {transmission[i0]:.3f}")
    print(f"Clean-strip modes at E=0: N = {clean_transmission[i0]}")
    print(f"Texture-induced loss at E=0: Delta T = {clean_transmission[i0] - transmission[i0]:.3f}")
    T_min_idx = np.argmin(transmission)
    print(f"Min T: {transmission[T_min_idx]:.3f} at E = {energies[T_min_idx]:.3f} t")
    print("At zero temperature, G(E) = (e^2/h) T(E).")
    print(f"\nFigure saved to: {output_path}")
