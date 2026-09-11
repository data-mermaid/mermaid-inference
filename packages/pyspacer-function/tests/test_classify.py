import ast
from pathlib import Path

import pyspacer_function.classify as classify_mod
from PIL import Image
from spacer.data_classes import ImageFeatures
from spacer.messages import DataLocation

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


def test_classify_legacy_scores_from_the_pickled_classifier(
    tmp_path, legacy_model_files, legacy_classes, fake_extractor_cls
):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), "white").save(img_path)
    image_loc = DataLocation("filesystem", str(img_path))
    points = [(10, 10), (20, 30)]
    vectors = {(10, 10): [0.9, 0.1, 0.2, 0.3], (20, 30): [0.1, 0.9, 0.3, 0.2]}

    results, valid = classify(
        image_loc, legacy_model_files, points, extractor=fake_extractor_cls(vectors)
    )

    assert valid is True
    assert [(pr.row, pr.col) for pr in results] == points
    # Each vector sits near a distinct trained centre, so the winning label
    # must match that centre specifically, not just belong to the label set.
    assert results[0].scores[0].label == legacy_classes[0]
    assert results[1].scores[0].label == legacy_classes[1]
    for pr in results:
        labels = [s.label for s in pr.scores]
        scores = [s.score for s in pr.scores]
        assert sorted(labels) == sorted(legacy_classes)  # full per-class list
        assert scores == sorted(scores, reverse=True)  # descending
        assert abs(sum(scores) - 1.0) < 1e-5  # calibrated probabilities


def test_classify_writes_features_that_round_trip_through_imagefeatures_load(
    tmp_path, model_files, fake_extractor_cls
):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), "white").save(img_path)
    image_loc = DataLocation("filesystem", str(img_path))
    points = [(10, 10), (20, 30)]
    vectors = {(10, 10): [3.0, 0.0, 0.0, 0.0], (20, 30): [0.0, 3.0, 0.0, 0.0]}
    feature_loc = DataLocation("filesystem", str(tmp_path / "out.featurevector"))

    classify(
        image_loc,
        model_files,
        points,
        extractor=fake_extractor_cls(vectors),
        feature_output_loc=feature_loc,
    )

    loaded = ImageFeatures.load(feature_loc)
    assert loaded.valid_rowcol is True
    assert [(pf.row, pf.col) for pf in loaded.point_features] == points
    for pf, (row, col) in zip(loaded.point_features, points):
        assert list(pf.data) == vectors[(row, col)]


def test_classify_writes_nothing_when_no_feature_output_loc(
    tmp_path, model_files, fake_extractor_cls, monkeypatch
):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), "white").save(img_path)
    image_loc = DataLocation("filesystem", str(img_path))
    points = [(10, 10)]
    vectors = {(10, 10): [3.0, 0.0, 0.0, 0.0]}

    store_calls = []
    monkeypatch.setattr(ImageFeatures, "store", lambda self, loc: store_calls.append(loc))

    classify(image_loc, model_files, points, extractor=fake_extractor_cls(vectors))

    assert store_calls == []


def test_classify_module_has_no_mermaid_classifier_import_at_module_scope():
    # The legacy image installs no mermaid-classifier, so a module-scope import
    # would break it on import.
    tree = ast.parse(Path(classify_mod.__file__).read_text())
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "mermaid_classifier"
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "mermaid_classifier"
