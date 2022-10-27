import os.path as osp

from setuptools import find_packages, setup


def get_version():
    # From: https://github.com/facebookresearch/iopath/blob/master/setup.py
    # Author: Facebook Research
    init_py_path = osp.join(osp.abspath(osp.dirname(__file__)), "greatx",
                            "version.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [
        line.strip() for line in init_py if line.startswith("__version__")
    ][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


VERSION = get_version()
url = 'https://github.com/EdisonLeeeee/GreatX'

install_requires = [
    'tqdm',
    'scipy',
    'numpy',
    'tabulate',
    'pandas',
    'termcolor',
    'scikit_learn',
    'matplotlib',
]

full_requires = [
    'numba',
    'pandas',
    'matplotlib',
    'networkx>=2.3',
    # 'gensim>=3.8.0',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

dev_requires = test_requires + [
    'pre-commit',
]

setup(
    name='greatx',
    version=VERSION,
    description='Graph reliability toolbox',
    author='Jintang Li',
    author_email='lijt55@mail2.sysu.edu.cn',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, VERSION),
    keywords=[
        'torch_geometric',
        'pytorch',
        'benchmark',
        'geometric-adversarial-learning',
        'graph-neural-networks',
    ],
    python_requires='>=3.7',
    license="MIT LICENSE",
    install_requires=install_requires,
    extras_require={
        'full': full_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    packages=find_packages(exclude=("examples", "imgs", "benchmark", "test")),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
