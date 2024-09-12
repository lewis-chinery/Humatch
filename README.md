# Humatch

**Fast, gene-specific joint humanisation of antibody heavy and light chains.**

<!--- INSTALL --->
## Install

Follow the steps below to download the code and the necessary packages. Requires **Python 3.9**.

```
# clone the repo
git clone https://github.com/oxpig/Humatch.git
cd Humatch/

# create your virtual env e.g.
python3 -m venv .humatch_venv
source .humatch_venv/bin/activate

# install
pip install .
```

**ANARCI** is required for aligning and padding sequences. We recommend installing ANARCI from <a href="https://github.com/oxpig/ANARCI/tree/master">github.com/oxpig/ANARCI</a>.

If you are having issues installing, try upgrading pip: *pip install --upgrade pip*.

Specific python versions can be used to initiate the environment using e.g. */usr/bin/python3.9 -m venv .humatch_venv*
