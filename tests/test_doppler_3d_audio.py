import pytest
import media.doppler_3d_audio as mod

CASES = [
    ("velocity_smooth", (1,)),
    ("crossing_guard", (1,)),
    ("attenuation_dedupe", (1,)),
    ("atomic_transform", (1,)),
    ("pause_history", (1,)),
    ("doppler_cap", (1,)),
    ("distance_model", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("air_absorb", (1,)),
    ("metrics", (1,)),
    ("listener_move", (1,)),
    ("mix_thread", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
