import numpy as np
from scipy.optimize import fsolve
import warnings


class CylinderBuckling:
    """
    Class for calculating cylinder buckling strengths based on DNV-RP-C202.

    Supports the following cylinder configurations:
    - Ring stiffened
    - Longitudinal stiffened
    - Orthogonally stiffened

    Currently supports calculations for:
    - Shell buckling
    """

    def __init__(
        self,
        l,
        d,
        t,
        ring_stiffeners=False,
        num_longitudinal=0,
        E=200e9,
        G=79.3e9,
        sigma_y=345e6,
        gamma=0.0,
        mod_length=0.0,
        **kwargs,
    ):
        """
        Creates an instance of `CylinderBuckling`.

        Parameters
        ----------
        z : np.array | list
            Section positions (m).
        d : np.array | list
            Diameters at cylinder points defined in L (m2).
        t : np.array | list
            Thicknesses at cylinder points defined in L (m2).
        ring_stiffeners : bool
            Toggle ring stiffeners at section changes.
            Default: False
        num_longitudinal : int
            Number of longitudinal stiffeners.
            Default: 0
        E : int | float
            Isotropic Young's modulus (Pa).
            Default: 200e9
        G : int | float
            Shear modulus (Pa).
            Default: 79.39e9
        sigma_y : int | float
            Yield Stress (Pa).
            Default: 345e6
        gamma : int | float
            Material partial safety factor.
        """

        self._l = np.array(l)
        self._d = np.array(d)
        self._t = np.array(t)
        self.E = E
        self.G = G
        self.sigma_y = sigma_y
        self.gamma = gamma
        self.mod_length = mod_length

        self.A = kwargs.get("A", np.zeros(len(self.t)))
        self.I = kwargs.get("I", np.zeros(len(self.t)))

        self._ring = ring_stiffeners
        self._long = num_longitudinal

    @property
    def d(self):
        """Section diameters at midpoints (m)."""
        if self._d.size == self._l.size + 1:
            return 0.5 * (self._d[1:] + self._d[:-1])
        elif self._d.size == self._l.size:
            return self._d
        else:
            raise ValueError(f"Incompatible size for diameter. Expected: {self._t.size + 1} vs {self._d.size}")

    @property
    def r(self):
        """Section radii at midpoints (m)."""

        return 0.5 * self.d

    @property
    def t(self):
        """Section thicknesses at midpoints (m)."""
        if self._l.size == self._t.size:
            return self._t
        elif self._l.size + 1 == self._t.size:
            return 0.5 * (self._t[1:] + self._t[:-1])
        else:
            raise ValueError(f"Incompatible size for thickness. Expected: {self._d.size - 1} vs {self._t.size}")

    @property
    def te(self):
        """Equivalent section thickness (m). Only applicable for longitudinal membrane stress."""

        if self.s is False:
            return self.t

        return self.t + self.A / self.s

    @property
    def Ac(self):
        """Cross sectional area of complete cylinder section"""
        Ac = (self.d**2 - (self.d - 2 * self.te) ** 2) * np.pi / 4.0
        Ac[self.A>0] = self.A[self.A>0]
        return Ac

    @property
    def Ic(self):
        """Cross sectional area of complete cylinder section"""
        Ic = (self.d**4 - (self.d - 2 * self.te) ** 4) * np.pi / 64.0
        Ic[self.I>0] = self.I[self.I>0]
        return Ic

    @property
    def s(self):
        """Section distance between longitudinal stiffeners (m)."""

        if self._long == 0:
            return False

        return np.pi * self.d / self._long

    @property
    def l(self):
        """Section heights (m)."""
        return self._l

    @property
    def v(self):
        """Poisson ratio for isotropic materials."""

        return 0.5 * self.E / self.G - 1.0

    @property
    def curvature_parameter(self):
        """"""

        if self._long:
            return ((self.s**2) / (self.r * self.t)) * np.sqrt(1 - self.v**2)

        else:
            return ((self.l**2) / (self.r * self.t)) * np.sqrt(1 - self.v**2)

    @property
    def Z(self):
        """Shorthand for `self.curvature_parameter`."""

        return self.curvature_parameter

    ## Buckling Coefficients
    # Ring Stiffened (unstiffened cylinder)
    def _C_axial_bending_unstiffened(self):
        """See Section 3.4, Table 3-2."""

        psi = 1
        xi = 0.702 * self.Z
        rho = 0.5 * (1 + self.r / (150 * self.t)) ** -0.5
        return psi * np.sqrt(1 + (rho * xi / psi) ** 2)

    def _C_torsion_unstiffened(self):
        """See Section 3.4, Table 3-2."""

        psi = 5.34
        xi = 0.856 * self.Z**0.75
        rho = 0.6
        return psi * np.sqrt(1 + (rho * xi / psi) ** 2)

    def _C_lateral_unstiffened(self):
        """See Section 3.4, Table 3-2."""

        psi = 4
        xi = 1.04 * np.sqrt(self.Z)
        rho = 0.6
        return psi * np.sqrt(1 + (rho * xi / psi) ** 2)

    def _C_hydrostatic_unstiffened(self):
        """See Section 3.4, Table 3-2."""

        psi = 2
        xi = 1.04 * np.sqrt(self.Z)
        rho = 0.6
        return psi * np.sqrt(1 + (rho * xi / psi) ** 2)

    # Longitudinal Stiffened and Orthongonally Stiffened
    def _C_axial_longitudinal(self):
        """See Section 3.3, Table 3-1."""

        if self.s is False:
            raise ValueError("Cylinder is not longitudinally stiffened. Incorrect use of this equation.")

        psi = 4
        xi = 0.702 * self.Z
        rho = 0.5 * (1 + self.r / (150.0 * self.t)) ** -0.5
        return psi * np.sqrt(1 + (rho * xi / psi) ** 2)

    def _C_shear_longitudinal(self):
        """See Section 3.3, Table 3-1."""

        if self.s is False:
            raise ValueError("Cylinder is not longitudinally stiffened. Incorrect use of this equation.")

        psi = 5.34 + 4 * (self.s / self.l) ** 2
        xi = 0.856 * np.sqrt(self.s / self.l) * self.Z**0.75
        rho = 0.6
        return psi * np.sqrt(1 + (rho * xi / psi) ** 2)

    def _C_compression_longitudinal(self):
        """See Section 3.3, Table 3-1."""

        if self.s is False:
            raise ValueError("Cylinder is not longitudinally stiffened. Incorrect use of this equation.")

        psi = (1 + (self.s / self.l) ** 2) ** 2
        xi = 1.04 * (self.s / self.l) * np.sqrt(self.Z)
        rho = 0.6
        return psi * np.sqrt(1 + (rho * xi / psi) ** 2)

    # Strength Calculations
    @property
    def fea(self):
        """Characteristic buckling strength due to axial or bending loads."""

        if self._long:
            return self._shell_buckling_strength(self._C_axial_longitudinal())

        else:
            return self._shell_buckling_strength(self._C_axial_bending_unstiffened())

    @property
    def fet(self):
        """Characteristic buckling strength due to torsion."""

        if self._long:
            return self._shell_buckling_strength(self._C_shear_longitudinal())

        else:
            return self._shell_buckling_strength(self._C_torsion_unstiffened())

    @property
    def feh(self):
        """Charcteristic buckling strength due to circumferntial compression."""

        if self._long:
            return self._shell_buckling_strength(self._C_compression_longitudinal())

        else:
            return self._shell_buckling_strength(self._C_lateral_unstiffened())

    def _shell_buckling_strength(self, C):
        """"""

        if self._long:
            return ((C * np.pi**2 * self.E) / (12 * (1 - self.v**2))) * (self.t / self.s) ** 2

        else:
            return ((C * np.pi**2 * self.E) / (12 * (1 - self.v**2))) * (self.t / self.l) ** 2

    def von_mises(self, axial, hoop, shear):
        a = ((axial + hoop) / 2.0) ** 2
        b = ((axial - hoop) / 2.0) ** 2
        c = shear**2

        return np.sqrt(a + 3.0 * (b + c))

    # Utilizations
    def run_buckling_checks(self, Fz, M, sigma_x, sigma_h, sigma_t):
        """

        Parameters
        ----------
        sigma_a :
        sigma_h :
        sigma_t :
        """

        # Axial stress contributions
        sigma_a = Fz / (2 * np.pi * self.r * self.te)
        sigma_m = M * self.r / self.Ic

        shell_utilization = self.utilization_shell(sigma_a, sigma_m, sigma_h, sigma_t)
        fak = self.get_fak(sigma_a, sigma_m, sigma_h, sigma_t)
        global_utilization = self.utilization_global(sigma_a, sigma_m, fak)

        return {"Shell": shell_utilization, "Global": global_utilization}

    def utilization_shell(self, _axial, _bending, _hoop, _shear):
        # Set Positive Stresses Equal to 0. See 3.2.4 - 3.2.6
        axial = np.abs(np.minimum(_axial, 0.0))
        # Bending comes in as a vector magnitude and is always positive
        bending = np.abs(_bending) #np.abs(np.minimum(_bending, 0.0))
        hoop = np.abs(np.minimum(_hoop, 0.0))
        shear = np.abs(_shear)
        
        eps = 1e-7
        vm = self.von_mises(axial, hoop, shear) + eps

        lambda_s = np.sqrt((self.sigma_y / vm) * ((axial + bending) / self.fea + shear / self.fet + hoop / self.feh))

        # Use DNV gamma unless specified by user
        gamma_m = 0.85 + 0.6 * lambda_s
        gamma_m[lambda_s < 0.5] = 1.15
        gamma_m[lambda_s >= 1.0] = 1.45
        gamma = self.gamma if self.gamma > 0.0 else gamma_m

        fks = self.sigma_y / np.sqrt(1 + lambda_s**4.0)
        fksd = fks / np.array(gamma)

        shell_util = vm / fksd

        return shell_util

    def get_fak(self, axial, bending, hoop, shear):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fak = fsolve(self._fak_wrapper, -np.abs(axial), args=(bending, hoop, shear), xtol=1e-4, maxfev=50)

        # Using mean instead of min here for more stable performance in optimization
        return np.abs(fak).mean()

    def _fak_wrapper(self, axial, *data):
        bending, hoop, shear = data
        util = self.utilization_shell(axial, bending, hoop, shear)

        return util - 1

    def utilization_global(self, _sigma_a, sigma_m, fak):
        """"""
        # This is column buckling strength in DNVGL
        # Stresses
        sigma_a = np.abs(np.minimum(_sigma_a, 0.0))
        # sigma_h = np.abs(np.minimum(_sigma_h, 0.0))

        # Euler buckling strength
        k = 2.0  # Fixed-free
        L = self._l.sum() - self.mod_length

        fE = (np.pi**2 * self.E * self.Ic) / ((k * L) ** 2 * self.Ac)

        # ls
        lambda_s = k * L / (np.pi * np.sqrt(self.Ic / self.Ac)) * np.sqrt(fak / self.E)

        # Use DNV gamma unless specified by user
        gamma_m = 0.85 + 0.6 * lambda_s
        gamma_m[lambda_s < 0.5] = 1.15
        gamma_m[lambda_s >= 1.0] = 1.45
        gamma = self.gamma if self.gamma > 0.0 else gamma_m

        # fak
        fakd = fak / gamma

        fkc = (1 - 0.28 * lambda_s**2)
        fkc[lambda_s > 1.34] = 0.9 / (lambda_s[lambda_s > 1.34] ** 2)
        fkc = fkc * fak
        fkcd = np.array(fkc / gamma_m)

        util = sigma_a / fkcd + 1 / fakd * (sigma_m / (1 - sigma_a / fE))
        return util
