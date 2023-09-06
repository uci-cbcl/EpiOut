import yaml
from epiout.annotation_config import download_ccres, create_config


def test_download_ccres(tmpdir):
    download_dir = tmpdir / "ccres"
    download_ccres(download_dir)

    assert (download_dir / 'ccres-ctcf-only.bed').exists()
    assert (download_dir / 'ccres-distal-enhancer.bed').exists()
    assert (download_dir / 'ccres-promoter.bed').exists()
    assert (download_dir / 'ccres-proximal-enhancer.bed').exists()


def test_create_config_motor_neuron(tmpdir):
    config_path = tmpdir / 'config.yaml'

    config = create_config('motor neuron', tmpdir, tf=True, ccres=True)

    assert config_path.exists()
    encode_ids = [
        'ENCFF071YEU',
        'ENCFF153WWO',
        'ENCFF294IIV',
        'ENCFF456GWH',
        'ENCFF535MNS',
        'ENCFF785LEY',
        'ENCFF817IVB',
        'ENCFF867WVM',
        'ENCFF991FFY',
        'ENCFF061ZQE',
        'ENCFF149XGC',
        'ENCFF284BCD',
        'ENCFF308PHN',
        'ENCFF518ICW',
        'ENCFF707MQP',
        'ENCFF788KZX',
        'ENCFF855BLF',
        'ENCFF916FML'
    ]

    for encode_id in encode_ids:
        assert (tmpdir / f'{encode_id}.bed.gz').exists()

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    assert config == {
        'H3K27ac': [
            'ENCFF817IVB.bed.gz',
            'ENCFF916FML.bed.gz',
            'ENCFF707MQP.bed.gz',
            'ENCFF294IIV.bed.gz',
            'ENCFF788KZX.bed.gz',
            'ENCFF518ICW.bed.gz'
        ],
        'H3K4me1': [
            'ENCFF456GWH.bed.gz',
            'ENCFF284BCD.bed.gz',
            'ENCFF855BLF.bed.gz',
            'ENCFF061ZQE.bed.gz',
            'ENCFF149XGC.bed.gz',
            'ENCFF785LEY.bed.gz'
        ],
        'H3K4me3': [
            'ENCFF867WVM.bed.gz',
            'ENCFF153WWO.bed.gz',
            'ENCFF535MNS.bed.gz',
            'ENCFF991FFY.bed.gz',
            'ENCFF071YEU.bed.gz',
            'ENCFF308PHN.bed.gz'
        ],
        'ccres-ctcf-only': ['ccres-ctcf-only.bed'],
        'ccres-distal-enhancer': ['ccres-distal-enhancer.bed'],
        'ccres-promoter': ['ccres-promoter.bed'],
        'ccres-proximal-enhancer': ['ccres-proximal-enhancer.bed']
    }


def test_create_config_panc1(tmpdir):
    config_path = tmpdir / 'config.yaml'

    config = create_config('Panc1', tmpdir, tf=True)

    assert config_path.exists()

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    assert config == {
        'CTCF': [
            'ENCFF382VLH.bed.gz',
            'ENCFF489ITF.bed.gz',
            'ENCFF504IUB.bed.gz'
        ],
        'H3K27ac': ['ENCFF079UIK.bed.gz'],
        'H3K27me3': ['ENCFF394AZZ.bed.gz'],
        'H3K36me3': ['ENCFF233BNK.bed.gz'],
        'H3K4me1': ['ENCFF895FOQ.bed.gz'],
        'H3K4me3': [
            'ENCFF103AWU.bed.gz',
            'ENCFF304CMG.bed.gz'
        ],
        'H3K9me3': ['ENCFF309WNP.bed.gz'],
        'POLR2AphosphoS5': [
            'ENCFF298EWL.bed.gz',
            'ENCFF389OXG.bed.gz',
            'ENCFF800POU.bed.gz'
        ],
        'REST': [
            'ENCFF200LXA.bed.gz',
            'ENCFF166KCQ.bed.gz',
            'ENCFF622LOQ.bed.gz',
            'ENCFF865TNO.bed.gz',
            'ENCFF960JTG.bed.gz',
            'ENCFF039IZP.bed.gz',
            'ENCFF087KDW.bed.gz',
            'ENCFF137ORU.bed.gz',
            'ENCFF705TKE.bed.gz'
        ],
        'SIN3A': [
            'ENCFF222OJN.bed.gz',
            'ENCFF515MOK.bed.gz',
            'ENCFF981YPH.bed.gz'
        ],
        'TCF7L2': [
            'ENCFF311JWR.bed.gz',
            'ENCFF990SDV.bed.gz',
            'ENCFF724WUN.bed.gz'
        ]
    }
