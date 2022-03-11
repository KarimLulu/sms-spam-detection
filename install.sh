#!/bin/bash
python3.8 -m venv --without-pip sms-env
source sms-env/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python

pip install -r requirements/common.txt