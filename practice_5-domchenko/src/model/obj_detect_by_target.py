from transformers import pipeline


# пайплайн с моделью
def get_obj_detect_by_target():
    return pipeline(
        task="zero-shot-object-detection",
        model="google/owlvit-base-patch32",
    )
