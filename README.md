Introduction to Radar Course
----------------------------

Installation
============

Welcome to the course material for the Introduction to Radar class. The main lectures come in the form of Jupyter notebooks with interactive elements. 
## Windows

### JupyterLab Installation

If you have not, first install Python on your system; you can download the Python installer from [here](https://www.python.org/downloads/).

Next, we will have to make a virtual environment to run JupyterLab and the radar course. A *virtual environment* is a minimal, isolated copy of Python that will keep the lab software from interfering with your other Python projects; you can read more about them [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment). 

To create our virtual environment, first create a folder of your choice, which will now be referred to as `<venv-dir>`. After creating the directory, open a command prompt and run:

    cd <venv-dir>
    py -m venv .

This will construct the Python virtual environment in this directory. Next, we will activate the virtual environment. Staying in the `<venv-dir>` directory, run:

    .\Scripts\activate

The command prompt should now show that you are in your virtual environment, i.e., if the virtual environment directory was `c:\test`, the command prompt should now look like

    (test) c:\test>

Next, we will install the necessary prerequisite packages. To do this, we use the Python installation routine `pip` by typing:

    pip install jupyterlab numpy scipy matplotlib ipympl jupyterlab-mathjax3

**Note**: If you are behind a proxy server at `<proxy-address>`, you will have to run `pip` using:

    pip install --proxy <proxy-address> jupyterlab numpy scipy matplotlib ipympl jupyterlab-mathjax3

### Radar Course

To start the labs, we first starting with new terminal/command prompt and activating the virtual environment by typing:

    cd <venv-dir>
    .\Scripts\activate

Navigate to the path of the radar course material:

    cd <radar-course-path>

Lastly, we can start JupyterLab in the current location:

    jupyter lab
    

## macOS

### JupyterLab Installation

If you have not, first install Python on your system; you can download the Python installer from [here](https://www.python.org/downloads/).

Next, we will have to make a virtual environment to run JupyterLab and the radar course. A *virtual environment* is a minimal, isolated copy of Python that will keep the lab software from interfering with your other Python projects; you can read more about them [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment). 

To create our virtual environment, first create a folder of your choice, which will now be referred to as `<venv-dir>`. After creating the directory, open a command prompt and run:

    cd <venv-dir>
    python3 -m venv .

This will construct the Python virtual environment in this directory. Next, we will activate the virtual environment. Staying in the `<venv-dir>` directory, run:

    source ./bin/activate

The terminal should now show that you are in your virtual environment, i.e., if the virtual environment directory was `~/test`, the command prompt should now look like

    (test) user@machine:~/test$

Next, we will install the necessary prerequisite packages. To do this, we use the Python installation routine `pip` by typing:

    pip install jupyterlab numpy scipy matplotlib ipympl jupyterlab-mathjax3

**Note**: If you are behind a proxy server at `<proxy-address>`, you will have to run `pip` using:

    pip install --proxy <proxy-address> jupyterlab numpy scipy matplotlib ipympl jupyterlab-mathjax3

### Radar Course

To start the labs, we first starting with new terminal/command prompt and type:

    cd <venv-dir>
    source ./bin/activate

Navigate to the path of the radar course material:

    cd <radar-course-path>

Lastly, we can start JupyterLab in the current location:

    jupyter lab
    
## Ubuntu

### JupyterLab Installation

If you have not, first install Python on your system. This can be done using `apt` by:

    sudo apt update
    sudo apt install python3 python3-venv

Next, we will have to make a virtual environment to run JupyterLab and the radar course. A *virtual environment* is a minimal, isolated copy of Python that will keep the lab software from interfering with your other Python projects; you can read more about them [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment). 

To create our virtual environment, first create a folder of your choice, which will now be referred to as `<venv-dir>`. After creating the directory, open a command prompt and run:

    cd <venv-dir>
    python3 -m venv .

This will construct the Python virtual environment in this directory. Next, we will activate the virtual environment. Staying in the `<venv-dir>` directory, run:

    source ./bin/activate

The terminal should now show that you are in your virtual environment, i.e., if the virtual environment directory was `~/test`, the command prompt should now look like

    (test) user@machine:~/test$

Next, we will install the necessary prerequisite packages. To do this, we use the Python installation routine `pip` by typing:

    pip install jupyterlab numpy scipy matplotlib ipympl jupyterlab-mathjax3

**Note**: If you are behind a proxy server at `<proxy-address>`, you will have to run `pip` using:

    pip install --proxy <proxy-address> jupyterlab numpy scipy matplotlib ipympl jupyterlab-mathjax3

### Radar Course

To start the labs, we first starting with new terminal/command prompt and type:

    cd <venv-dir>
    source ./bin/activate

Navigate to the path of the radar course material:

    cd <radar-course-path>

Lastly, we can start JupyterLab in the current location:

    jupyter lab

Authors
=======

Zachary Chance, Robert Freking, Victoria Helus

MIT Lincoln Laboratory
Lexington, MA 02421

Distribution Statement
======================

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the United States Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force.

© 2021 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

RAMS ID: 1016938