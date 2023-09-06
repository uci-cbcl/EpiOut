import pytest
from epiout.hic import HicReader


@pytest.fixture
def hic_reader():
    return HicReader(
        ["tests/data/test_chr22.hic", "tests/data/test_chr22.hic"],
        "tests/data/chrom.sizes",
    )


def test_HicReader_contact_score(hic_reader):
    scores = hic_reader.contact_scores('chr22')
    assert scores.shape == (10200, 401)
