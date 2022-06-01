## M1 GPU support (not working)
Not yet working as the vision/text library is not yet supported on m1 gpu yet. 

1. check that platform is _macOS-12.3-arm64-arm-64bit_ with:
        import platform
        platform.platform()

2. create a conda evironment for the ARM platform:
        CONDA_SUBDIR=osx-arm64 conda create -n ml python=3.9 -c conda-forge

3. activate the conda environment:
        conda activate ml

4. permenantly set the platform environment variable:
        conda env config vars set CONDA_SUBDIR=osx-arm64

5. reactive the conda environment:
        conda activate
        conda activate ml

6. install the pytorch nightly build that supports mac M1 GPU
        pip3 install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

7. install rust as some models are optimized with rust:
        curl — proto ‘=https’ — tlsv1.2 -sSf https://sh.rustup.rs | sh

8. install the requirements for mac:
        pip3 install -r requirements/requirements_mac_gpu.txt


## Notes
Since some we use the mac nightly build of torch for gpu support, it does not work with for example torchtext. You will get a 'Symbol not found' error.
https://discuss.pytorch.org/t/mps-does-not-work-with-projects-using-torchtext/152036
"This is indeed “expected” today: domain libraries do not provided nightly builds for arm. So you are getting the latest released version which is not binary compatible with a nightly version of torch.

As of right now, you can build torchtext from source to solve this.
We are also working on making these builds available (you can track Add TorchVision wheel build for M1 by malfet · Pull Request #5948 · pytorch/vision · GitHub for torchvision and we’re working on similar PRs for text and audio).
"

solution: Build TorchVision from source with the same conda env you used to isntall torch nightly.
        git clone https://github.com/pytorch/text torchtext
        cd torchtext
        git submodule update --init --recursive
        MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py clean install