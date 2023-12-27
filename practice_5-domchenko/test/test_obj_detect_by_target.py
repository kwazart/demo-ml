from transformers import pipeline

from src.model.obj_detect_by_target import get_obj_detect_by_target


def test_get_obj_detect_by_target():
    obj_detect_pip = get_obj_detect_by_target()
    assert obj_detect_pip is not None
    assert isinstance(obj_detect_pip, type(pipeline(
        task="zero-shot-object-detection",
        model="google/owlvit-base-patch32",
    )))
