import json
import stages


def test_all_expected_stages_registered_and_serializable():
    ids = {s["id"] for s in stages.list_stages()}
    expected = {"crop", "background_extract", "color_calibrate", "stretch", "finish",
                "saturate", "masked_denoise", "unsharp_luma", "remove_stars",
                "screen_recombine", "export_image", "preview_sink"}
    assert expected <= ids
    json.dumps(stages.list_stages())     # every schema is JSON-serializable
