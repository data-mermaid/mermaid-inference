from PIL import Image
from spacer.data_classes import DataLocation

from pyspacer_function.classify import classify


def test_classify_returns_full_sorted_scores(tmp_path, model_files, fake_extractor_cls):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), "white").save(img_path)
    image_loc = DataLocation("filesystem", str(img_path))
    points = [(10, 10), (20, 30)]
    vectors = {(10, 10): [3.0, 0.0, 0.0, 0.0], (20, 30): [0.0, 3.0, 0.0, 0.0]}

    results, valid = classify(
        image_loc, model_files, points, extractor=fake_extractor_cls(vectors)
    )

    assert valid is True
    assert len(results) == 2
    for pr in results:
        labels = [s.label for s in pr.scores]
        scores = [s.score for s in pr.scores]
        assert set(labels) == {"a::", "b::", "c::"}      # full per-class list
        assert scores == sorted(scores, reverse=True)     # descending
        assert abs(sum(scores) - 1.0) < 1e-5              # softmax probabilities
    assert (results[0].row, results[0].col) == (10, 10)
