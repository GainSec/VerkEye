from pathlib import Path


SOURCE = Path("src/verkeye/compat/native/ades_operator_capture.cpp")


def test_operator_capture_exports_each_ades_node_output_after_execution() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "node_base::copy_op_output" in source
    assert "write_data_to_file" in source
    assert "VERKEYE_OPERATOR_DUMP" in source
    assert 'resolve_loaded<WriteData>("libvamba_vec.so"' in source
    assert "node_fastconv_op4exec" in source
    assert "node_madd_op4exec" in source
    assert "node_trans_op4exec" in source
    assert "node_shuffle_op4exec" in source
    assert "node_lvl_curve_op4exec" in source
    assert "original(self);" in source
    assert "dump_outputs(self" in source


def test_operator_capture_exports_imported_vectors_before_execution() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "dag_base::import_vector" in source
    assert "node_mux_op::set_data" in source
    assert "VERKEYE_MASK_DUMP" in source
    assert "import-" in source
    assert "-slot-" in source
    assert "write_data(vector, output)" in source


def test_operator_capture_uses_the_recovered_pair_abi_without_guessing_values() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "kOutputPairSize = 0x158" in source
    assert "kVectorPointerOffset = 0x150" in source
    assert "static_assert(sizeof(RawVector) == 3 * sizeof(void*))" in source
    assert "context-specific" not in source
