#!/usr/bin/env bash

virtualenv --no-site-packages  -p /usr/bin/python3.5 venv
venv/bin/pip3 install --upgrade pip
venv/bin/pip3 install -r requirements.txt