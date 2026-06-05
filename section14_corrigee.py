# ======================================================================
# §14 — Reynolds comme résidu : somme des autres termes de Navier-Stokes
# CORRECTIONS :
#   1. dt_sec = 6h = 21600 s (pas de temps de sortie connu)
#   2. On travaille sur le vent MOYEN horizontal à chaque instant,
#      conformément à l'équation du vent moyen de Reynolds :
#        ∂ū/∂t = -ū·∂ū/∂x - ... - (1/ρ₀)·∂z(ρ₀·ū'w̄')
#      Tous les termes sont donc d'abord moyennés sur (y,x), puis
#      on accumule sur t → on compare bien à tend_u_glob du §7.
# ======================================================================

dt_sec = 6.0 * 3600.0   # 6 h en secondes
print(f'dt_sec = {dt_sec:.0f} s ({dt_sec/3600:.1f} h)')

residu = np.zeros(n_z)

ds_u = xr.open_dataset(path3d('ua'))
ds_v = xr.open_dataset(path3d('va'))
ds_w = xr.open_dataset(path3d('wa'))
ds_p = xr.open_dataset(path3d('pa'))

for iz in range(n_z):
    acc = 0.0
    n   = 0

    iz_lo = max(iz - 1, 0)
    iz_hi = min(iz + 1, n_z - 1)
    dz    = alt[iz_hi] - alt[iz_lo]
    if dz == 0:
        dz = alt[1] - alt[0]   # sécurité aux bords

    for it in range(t_stat, n_t - 1):

        # ── Champs 2D instantanés ──────────────────────────────────────
        u_2d = ds_u['ua'].isel({dim_t: it,   dim_z: iz}).values   # (ny, nx)
        v_2d = ds_v['va'].isel({dim_t: it,   dim_z: iz}).values
        w_2d = ds_w['wa'].isel({dim_t: it,   dim_z: iz}).values
        p_2d = ds_p['pa'].isel({dim_t: it,   dim_z: iz}).values

        # ── Moyennes horizontales (scalaires) ─────────────────────────
        u_bar   = u_2d.mean()
        v_bar   = v_2d.mean()
        w_bar   = w_2d.mean()

        # ── ∂ū/∂t  (différence finie temporelle sur le vent moyen) ────
        u_next  = ds_u['ua'].isel({dim_t: it + 1, dim_z: iz}).values
        u_bar_next = u_next.mean()
        dudt    = (u_bar_next - u_bar) / dt_sec
        del u_next

        # ── Advection du vent moyen par le vent moyen ─────────────────
        # Termes horizontaux : gradient de ū moyenné spatialement
        dudx_bar = np.gradient(u_2d, dx, axis=1).mean()
        dudy_bar = np.gradient(u_2d, dy, axis=0).mean()

        # Terme vertical : gradient centré de ū entre niveaux adjacents
        u_lo_bar = ds_u['ua'].isel({dim_t: it, dim_z: iz_lo}).values.mean()
        u_hi_bar = ds_u['ua'].isel({dim_t: it, dim_z: iz_hi}).values.mean()
        dudz_bar = (u_hi_bar - u_lo_bar) / dz
        del u_lo_bar, u_hi_bar

        adv = -(u_bar * dudx_bar + v_bar * dudy_bar + w_bar * dudz_bar)

        # ── Terme de pression (gradient horizontal de p̄) ─────────────
        pres = -np.gradient(p_2d, dx, axis=1).mean() / rho0[iz]

        acc += dudt + adv + pres
        n   += 1

        del u_2d, v_2d, w_2d, p_2d
        gc.collect()

    residu[iz] = acc / n if n > 0 else 0.0

    if iz % 10 == 0:
        print(f'  iz={iz}/{n_z-1}  ({alt[iz]:.0f} m)')

ds_u.close() ; ds_v.close() ; ds_w.close() ; ds_p.close()
del ds_u, ds_v, ds_w, ds_p
gc.collect()
print('Résidu calculé.')

# ======================================================================
# Visualisation
# ======================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(tend_u_glob * 1e5, alt, color='royalblue', lw=2,
        label=r"Reynolds direct $-\frac{1}{\rho_0}\partial_z(\rho_0\overline{u'w'})$")
ax.plot(residu * 1e5,      alt, color='crimson',   lw=2, linestyle='--',
        label='Résidu (∂ū/∂t + advection + pression)')
ax.axvline(0, color='grey', alpha=0.4)
ax.set_xlabel('Tendance (×10⁻⁵ m/s²)')
ax.set_ylabel('Altitude (m)')
ax.set_title('Test de fermeture du bilan de QdM', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
