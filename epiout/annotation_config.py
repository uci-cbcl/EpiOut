from collections import defaultdict
from pkg_resources import resource_filename
import yaml
import pooch
import pandas as pd
from pathlib import Path


ccres_urls = {
    'ccres-promoter': (
        'https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.PLS.bed',
        '0df08ff8483bf48a739f3f32d0355b59'
    ),
    'ccres-proximal-enhancer': (
        'https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.pELS.bed',
        'e36f3f284d77e778fe6d7407a4fd6fc0'
    ),
    'ccres-distal-enhancer': (
        'https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.dELS.bed',
        '91e232bb2100a8ea8a36fe4f17481c8a'
    ),
    'ccres-ctcf-only': (
        'https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.CTCF-only.bed',
        'f311596e2ed0c59d2d79e762cc625731'
    )
}

_encode_bed_url = 'https://www.encodeproject.org/files/{encode_id}/@@download/{encode_id}.bed.gz'
_encode_hic_url = 'https://www.encodeproject.org/files/{encode_id}/@@download/{encode_id}.hic'


def download_ccres(output_dir):
    '''
    Download cCREs from https://wenglab.org/registry/
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for name, (url, md5) in ccres_urls.items():
        pooch.retrieve(
            url=url,
            known_hash=f'md5:{md5}',
            path=output_dir, fname=f'{name}.bed'
        )


def _read(path):
    return pd.read_csv(
        resource_filename('epiout', path),
        index_col='tissue')


def read_hic():
    '''Read config file for Hi-C'''
    return _read('configs/epiannot_hic.csv.gz')


def read_histone():
    '''Read config file for histone chipseq experiments.'''
    return _read('configs/epiannot_histone.csv.gz')


def read_tf():
    '''Read config file for TF chipseq experiments.'''
    return _read('configs/epiannot_tf.csv.gz')


def create_config(tissue, output_dir, tf=False, hic=False, ccres=False):
    '''
    Create config file for annotation for given list of cell types

    Args:
      tissue: list of cell types or tissues to include in the config file.
      output_dir: output directory where the config file will be saved 
        and the annotation files will be downloaded.
      hic: whether include hic files (takes a very long time to download)
      ccres: whether include ccres files contains pre-defiend promoter, 
        proximal enhancer, distal enhancers.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # print
    df_histone = read_histone()

    if (df_histone.index == tissue).any():
        df_histone = df_histone.loc[tissue]
        print('Downloading following histone files:')
        print(df_histone)

    if tf:
        df_tf = read_tf()
        if (df_tf.index == tissue).any():
            df_tf = df_tf.loc[tissue]
            print('Downloading following tf files:')
            print(df_tf)
        else:
            tf = False

    if hic:
        df_hic = read_hic()
        if (df_hic.index == tissue).any():
            df_hic = df_hic.loc[tissue]
            print('Downloading following hic files:')
            print(df_hic)
        else:
            hic = False

    # Download
    files = defaultdict(list)

    if (df_histone.index == tissue).any():
        df_histone = df_histone.loc[tissue]

        for row in df_histone.itertuples():
            fname = f'{row.encode_id}.bed.gz'
            pooch.retrieve(
                url=_encode_bed_url.format(encode_id=row.encode_id),
                known_hash=f'md5:{row.md5sum}',
                path=output_dir, fname=fname
            )
            files[row.chipseq].append(fname)

    if tf:
        for row in df_tf.itertuples():
            fname = f'{row.encode_id}.bed.gz'
            pooch.retrieve(
                url=_encode_bed_url.format(encode_id=row.encode_id),
                known_hash=f'md5:{row.md5sum}',
                path=output_dir, fname=fname
            )
            files[row.chipseq].append(fname)

    if hic:
        for row in df_hic.itertuples():
            fname = f'{row.encode_id}.hic'
            pooch.retrieve(
                url=_encode_hic_url.format(encode_id=row.encode_id),
                known_hash=f'md5:{row.md5sum}',
                path=output_dir, fname=fname,
                progressbar=True
            )
            files['hic'].append(fname)

    if ccres:
        download_ccres(output_dir)
        for name in ccres_urls:
            files[name].append(f'{name}.bed')

    return AnnotationConfig.from_dict(dict(files), output_dir)


class AnnotationConfig:
    '''
    Config file for annotation files.

    Args:
      config_path: path to the config file.
    '''

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.config = self._read()

    def _read(self):
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        self._validate(config)

        for name, files in config.items():
            for i, f in enumerate(files):
                config[name][i] = str(self.config_dir / f)

        return config

    def _validate(self, config):
        for name, files in config.items():
            for f in files:
                if name == "hic":
                    assert f.endswith(
                        ".hic"), "Only .hic file supported for hic"
                else:
                    assert f.endswith(".bed") or f.endswith(
                        ".bed.gz"
                    ), "Only .bed file format is supported"

    def to_dict(self):
        return self.config

    @classmethod
    def from_dict(cls, config, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        config_path = output_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return cls(config_path)
