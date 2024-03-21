from practice_3.src.transformers.image_to_text import load_model


def test_image_to_text():
    image_to_text = load_model()
    assert callable(image_to_text)
