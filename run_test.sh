#!/bin/bash

pytest cv19index/ -s --log-level=DEBUG --ignore=python --junitxml=pytest.xml || true