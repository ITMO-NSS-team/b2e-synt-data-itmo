"""Regression coverage for JSON-round-tripped mode conditions on resume."""
from copy import deepcopy
import pytest
from sim.benchmark.results import ResultWriter


def test_resume_accepts_tuple_conditions_after_json_round_trip(tmp_path):
    manifest = {
        "eval_id": "resume-json", "workers": 4, "repetitions": 3,
        "case_ids": ["case-0270-00"], "case_hashes": {"case-0270-00": "case-hash"},
        "live_stand": {"skill_registry_hash": "registry-hash"},
        "modes": [{"name": "generated_skills@compact", "catalog_hash": "catalog-hash",
                   "tool_subset": ("find_skills", "mcp_query"),
                   "generated_skill_names": ("salary-review",),
                   "common": {"hr_employee_ids": ("9877478",), "temperature": 0.0}}],
    }
    original = ResultWriter(tmp_path, "resume-json").write_manifest(manifest)
    saved_bytes = original.read_bytes()
    resumed = ResultWriter(tmp_path, "resume-json", resume=True)
    assert resumed.write_manifest(manifest) == original
    assert original.read_bytes() == saved_bytes
    for field, value in [("catalog_hash", "different-catalog"), ("tool_subset", ("mcp_query",))]:
        changed = deepcopy(manifest)
        changed["modes"][0][field] = value
        with pytest.raises(ValueError, match="resume manifest differs"):
            resumed.write_manifest(changed)
    changed = deepcopy(manifest)
    changed["modes"][0]["common"]["hr_employee_ids"] = ("other-hr",)
    with pytest.raises(ValueError, match="resume manifest differs"):
        resumed.write_manifest(changed)
