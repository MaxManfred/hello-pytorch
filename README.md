# stai-time-expression-normalizer

## Description 

A library for detecting temporal expressions in unstructured text and normalizing them into a time range. It is based on the CoreNLP SUTime java library (see https://nlp.stanford.edu/software/sutime.shtml)

## Setup
Add additional setup instructions & requirements here

## How to use

Add usage example here:
```
# usage example
```


## License
Copyright (c) Konica Minolta Business Solutions. All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
Written by Massimo Manfredino <massimo.manfredino@konicaminolta.it>

## Project documentation

### Install package devel:
**Warning**: Always create new Python virtual environment before installing development mode packages
! (use https://virtualenvwrapper.readthedocs.io/en/latest/).
```
pip install -e .[test]
```

### Install package:

```
pip install .
```

### Run tests:

```
py.test
```

### Increase version of package:

#### 1. change version in ```setup.py```

Modify the ```__version__``` variable.

#### 2. Commit your changes
```
git commit -am '<COMMIT MESSAGE>'
```
#### 3. Add new git tag

Tag has to be the same as the version in ```setup.py```

```
git tag <version>
git push
git push --tags
```

### Upload to PIP server

If requested for password, do not type in any, just hit ENTER.

```
python setup.py sdist upload -r local
```

### Install package with PIP:

```
pip install stai-time-expression-normalizer
```

### Upgrade package with PIP:

```
pip install --upgrade stai-time-expression-normalizer
```

### How to setup PIP server access:

https://bitbucket.kmiservicehub.com/projects/KR/repos/stai-pypi-server/

### How to build jpy wheel on Ubuntu

```
1. Install Python 3.6, Oracle JDK 8 and Maven 3
3. Install Python Development Tools 
	$ sudo apt-get install python3-dev    
2. Set JAVA_HOME and JDK_HOME to point to your JDK installation
	$ export JDK_HOME=<your-jdk-dir>
	$ export JAVA_HOME=$JDK_HOME
3. Clone jpy project from Git repository https://github.com/bcdev/jpy
3. Run this commands in the jpy folder
	$ python3 setup.py --maven clean
	$ python3 get-pip.py (or sudo -H python3 get-pip.py)
	$ python3 setup.py --maven bdist_wheel
4. On success, the wheel is found in the dist directory
```

### How to upload jpy wheel on company PyPI server

```
$ pip3 install twine
$ twine upload wheel_file_name.whl --repository-url http://rd-brn-ch-h2.bic.local:9292
$ username = type "stai"
$ password = leave it blank
```# hello-pytorch
