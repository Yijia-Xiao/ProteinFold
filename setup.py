# Author: Yijia Xiao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


with open("fold/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()


setup(
    name="proteinfold",
    version=version,
    description="Large-scale pretrained protein MSA language models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Yijia Xiao",
    url="https://github.com/Yijia-Xiao/ProteinFold",
    license="MIT",
    packages=["fold"],
    data_files=[("source_docs/fold", ["LICENSE", "README.md", "CODE_OF_CONDUCT.rst"])],
    zip_safe=True,
)
