import os

import numpy as np
import openmdao.api as om
from scipy.interpolate import PchipInterpolator


class ParametrizeBladeAero(om.ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the outer shape of the wind turbine rotor blades
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.opt_options = self.options["opt_options"]
        n_span = rotorse_options["n_span"]
        self.n_opt_twist = n_opt_twist = self.opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]
        self.n_opt_chord = n_opt_chord = self.opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"]

        # Inputs
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        # # Blade twist
        # self.add_input(
        #     "twist_original",
        #     val=np.zeros(n_span),
        #     units="rad",
        #     desc="1D array of the twist values defined along blade span. The twist is the one defined in the yaml.",
        # )
        self.add_input(
            "s_opt_twist",
            val=np.zeros(n_opt_twist),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist angle",
        )
        self.add_input(
            "twist_opt",
            val=np.ones(n_opt_twist),
            units="rad",
            desc="1D array of the twist angle being optimized at the n_opt locations.",
        )
        self.add_input(
            "chord_original",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the chord values defined along blade span. The chord is the one defined in the yaml.",
        )
        self.add_input(
            "s_opt_chord",
            val=np.zeros(n_opt_chord),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade chord",
        )
        self.add_input(
            "chord_opt",
            val=np.ones(n_opt_chord),
            units="m",
            desc="1D array of the chord being optimized at the n_opt locations.",
        )
        self.add_input(
            "section_offset_y",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the airfoil position relative to the reference axis, specifying the distance in meters along the chordline from the reference axis to the leading edge. The distribution is the original from the yaml.",
        )
        self.add_input(
            "ac_interp",
            val=np.zeros(n_span),
            desc="1D array of the aerodynamic center of the blade defined along span.",
        )
        # Outputs
        self.add_output(
            "twist_param",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is the result of the parameterization.",
        )
        self.add_output(
            "chord_param",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the chord values defined along blade span. The chord is the result of the parameterization.",
        )
        self.add_output(
            "section_offset_y_param",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the airfoil position relative to the reference axis, specifying the distance in meters along the chordline from the reference axis to the leading edge. The distribution is the result of the parameterization.",
        )
        self.add_output(
            "max_chord_constr",
            val=np.zeros(n_opt_chord),
            desc="1D array of the ratio between chord values and maximum chord along blade span.",
        )
        self.add_output(
            "slope_chord_constr",
            val=np.zeros(n_opt_chord-1),
            desc="1D array of the difference between one chord point and the other. It can be used as constraint to achieve monotically increasing and then decreasing chord",
        )
        self.add_output(
            "slope_twist_constr",
            val=np.zeros(n_opt_twist-1),
            desc="1D array of the difference between one twist point and the other. It can be used as constraint to achieve monotically decreasing and then increasing chord",
        )

    def compute(self, inputs, outputs):
        spline = PchipInterpolator
        twist_spline = spline(inputs["s_opt_twist"], inputs["twist_opt"])
        outputs["twist_param"] = twist_spline(inputs["s"])
        chord_spline = spline(inputs["s_opt_chord"], inputs["chord_opt"])
        outputs["chord_param"] = chord_spline(inputs["s"])
        chord_opt = spline(inputs["s"], outputs["chord_param"])
        max_chord = self.opt_options["constraints"]["blade"]["chord"]["max"]
        outputs["max_chord_constr"] = chord_opt(inputs["s_opt_chord"]) / max_chord
        # Define constraint to enforce monothonically increasing and then decreasing blade chord
        # Constraint is satisfied when below 0
        id_max_chord = np.argmax(inputs["chord_opt"])
        slope_chord_constr = np.diff(inputs["chord_opt"])
        # Up to max chord, chord must be increasing (positive diff), after max chord, decreasing (negative diff)
        slope_chord_constr[:id_max_chord] *= -1 
        outputs["slope_chord_constr"] = slope_chord_constr
        # Similarly, define constraint to enforce monothonically decreasing and then increasing blade twist
        id_min_twist = np.argmin(inputs["twist_opt"])
        slope_twist_constr = np.diff(inputs["twist_opt"])
        slope_twist_constr[id_min_twist:] *= -1 
        outputs["slope_twist_constr"] = slope_twist_constr
        # Update section_offset_y
        outputs["section_offset_y_param"] = inputs["section_offset_y"] * outputs["chord_param"] / inputs["chord_original"]
        # Alternative approach following the aerodynamic center, which probably needs some smoothing first though
        # outputs["section_offset_y_param"] = inputs["ac_interp"] * outputs["chord_param"]


class ParametrizeBladeStruct(om.ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the structural design of the wind turbine rotor blades
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.opt_options = opt_options = self.options["opt_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        # Inputs
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "layer_thickness_original",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )

        for i in range(n_layers):
            self.add_input(
                "s_opt_layer_%d"%i,
                val=np.ones(opt_options["design_variables"]["blade"]["n_opt_struct"][i]),
            )
            self.add_input(
                "layer_%d_opt"%i,
                units="m",
                val=np.ones(opt_options["design_variables"]["blade"]["n_opt_struct"][i]),
            )

        # Outputs
        self.add_output(
            "layer_thickness_param",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure after the parametrization. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )

    def compute(self, inputs, outputs):

        for i in range(self.n_layers):
            opt_m_interp = PchipInterpolator(inputs["s_opt_layer_%d"%i], inputs["layer_%d_opt"%i])(inputs["s"])
            outputs["layer_thickness_param"][i, :] = opt_m_interp

class ComputeReynolds(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_span")

    def setup(self):
        n_span = self.options["n_span"]

        self.add_input("rho", val=0.0, units="kg/m**3")
        self.add_input("mu", val=1.81e-5, units="kg/(m*s)", desc="Dynamic viscosity of air")
        self.add_input("chord", val=np.zeros((n_span)), units="m")
        self.add_input("r_blade", val=np.zeros(n_span), units="m",
            desc="1D array of the dimensional spanwise grid defined along the rotor (hub radius to blade tip projected on the plane)",
        )
        self.add_input("rotor_diameter", val=0.0, units="m",
            desc="Diameter of the wind turbine rotor specified by the user, defined as 2 x (Rhub + blade length along z) * cos(precone).",
        )
        self.add_input("maxOmega", val=0.0, units="rad/s", desc="Maximum allowed rotor speed.")
        self.add_input("V_out", val=0.0, units="m/s", desc="Cut out wind speed. This is the wind speed where region III ends.")

        self.add_output("Re", val=np.zeros((n_span)), ref=1.0e6)
        
    def compute(self, inputs, outputs):
        # Note that we used to use ccblade outputs of local wind speed at the rated condition
        # This is more accurate, of course, but creates an implicit feedback loop in the code
        # This way gets an order-of-magnitude estimate for Reynolds number, which is really all that is needed

        max_local_TS = inputs["maxOmega"] * inputs["r_blade"]
        max_local_V = np.sqrt(inputs["V_out"]**2 + max_local_TS**2)
        outputs["Re"] = np.nan_to_num(
            inputs["rho"] * max_local_V * inputs["chord"] / inputs["mu"]
        )
