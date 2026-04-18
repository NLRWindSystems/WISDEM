import shutil
import unittest
from pathlib import Path

import pytest

import wisdem.inputs as sch
from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = Path(__file__).parents[3] / "examples" / "02_reference_turbines"
fname_analysis_options = test_dir / "analysis_options.yaml"


@pytest.fixture(scope="function")
def no_wombat_model_file(tmp_path_factory, model_file):
    """Create a temporary second configuration without the OpEx flag set to False."""
    temp_dir = tmp_path_factory.mktemp("temp_dir")
    modeling_options = sch.load_modeling_yaml(test_dir / model_file)
    new_model_file = Path(model_file)
    new_model_file = temp_dir / new_model_file.with_stem(Path(model_file).stem + "_no_wombat")
    modeling_options["WISDEM"]["OpEx"] = {"flag": False}
    new_model_file = sch.write_modeling_yaml(modeling_options, str(new_model_file))
    yield new_model_file
    shutil.rmtree(str(temp_dir))


def test_5MW():
    """NREL 5 MW turbine test."""
    fname_wt_input = test_dir / "nrel5mw.yaml"
    fname_modeling_options = test_dir / "modeling_options_nrel5.yaml"

    wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

    assert wt_opt["rotorse.rp.AEP"][0] * 1.0e-6 == pytest.approx(23.8931681739, abs=0.01)
    assert wt_opt["rotorse.blade_mass"][0] == pytest.approx(
        16485.0072740210, abs=0.01
    )  # new value: improved interpolation
    assert wt_opt["financese.lcoe"][0] * 1.0e3 == pytest.approx(54.07483619479791, abs=0.1)
    assert wt_opt["rotorse.rs.tip_pos.tip_deflection"][0] == pytest.approx(4.4785104986, abs=0.1)
    assert wt_opt["towerse.z_param"][-1] == pytest.approx(87.7, abs=0.01)


@pytest.mark.parametrize("model_file", ["modeling_options_iea15.yaml"])
def test_15MW(no_wombat_model_file, subtests):
    """IEA 15 MW turbine test."""
    fname_wt_input = test_dir / "IEA-15-240-RWT.yaml"
    fname_modeling_options = test_dir / "modeling_options_iea15.yaml"
    wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

    with subtests.test("Check WOMBAT-based OpEx effects on LCOE"):
        assert wt_opt["rotorse.rp.AEP"][0] * 1.0e-6 == pytest.approx(77.90013659314998, abs=0.1)
        assert wt_opt["rotorse.blade_mass"][0] == pytest.approx(
            68233.0936092383, abs=1
        )  # new value: improved interpolation
        assert wt_opt["financese.lcoe"][0] * 1.0e3 == pytest.approx(73.16050426430054, abs=1.0)
        assert wt_opt["rotorse.rs.tip_pos.tip_deflection"][0] == pytest.approx(25.98145796253223, abs=0.1)
        assert wt_opt["towerse.z_param"][-1] == pytest.approx(144.386, abs=0.001)

    wt_opt, _, _ = run_wisdem(fname_wt_input, no_wombat_model_file, fname_analysis_options)
    with subtests.test("Check fixed assumption OpEx effects on LCOE"):
        assert wt_opt["rotorse.rp.AEP"][0] * 1.0e-6 == pytest.approx(77.90013659314998, abs=0.1)
        assert wt_opt["rotorse.blade_mass"][0] == pytest.approx(
            68233.0936092383, abs=1
        )  # new value: improved interpolation
        assert wt_opt["financese.lcoe"][0] * 1.0e3 == pytest.approx(76.07140864323416, abs=1.0)
        assert wt_opt["rotorse.rs.tip_pos.tip_deflection"][0] == pytest.approx(25.98145796253223, abs=0.1)
        assert wt_opt["towerse.z_param"][-1] == pytest.approx(144.386, abs=0.001)


@pytest.mark.parametrize("model_file", ["modeling_options_iea3p4.yaml"])
def test_3p4MW(no_wombat_model_file, subtests):
    """IEA 3.4 MW turbine test."""
    fname_wt_input = test_dir / "IEA-3p4-130-RWT.yaml"
    fname_modeling_options = test_dir / "modeling_options_iea3p4.yaml"
    wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

    with subtests.test("Check WOMBAT-based OpEx effects on LCOE"):
        assert wt_opt["rotorse.rp.AEP"][0] * 1.0e-6 == pytest.approx(13.591140338759166, abs=0.1)
        assert wt_opt["rotorse.blade_mass"][0] == pytest.approx(
            14534.711602944584, abs=0.1
        )  # new value: improved interpolation
        assert wt_opt["financese.lcoe"][0] * 1.0e3 == pytest.approx(36.66569502755669, abs=0.1)
        assert wt_opt["rotorse.rs.tip_pos.tip_deflection"][0] == pytest.approx(8.031667548036724, abs=0.1)
        assert wt_opt["towerse.z_param"][-1] == pytest.approx(108.0, abs=0.001)

    wt_opt, _, _ = run_wisdem(fname_wt_input, no_wombat_model_file, fname_analysis_options)
    with subtests.test("Check fixed assumption OpEx effects on LCOE"):
        assert wt_opt["rotorse.rp.AEP"][0] * 1.0e-6 == pytest.approx(13.591140338759166, abs=0.1)
        assert wt_opt["rotorse.blade_mass"][0] == pytest.approx(
            14534.711602944584, abs=0.1
        )  # new value: improved interpolation
        assert wt_opt["financese.lcoe"][0] * 1.0e3 == pytest.approx(39.62144631809757, abs=0.1)
        assert wt_opt["rotorse.rs.tip_pos.tip_deflection"][0] == pytest.approx(8.031667548036724, abs=0.1)
        assert wt_opt["towerse.z_param"][-1] == pytest.approx(108.0, abs=0.001)
