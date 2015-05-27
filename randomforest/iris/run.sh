#!/usr/bin/env bash
rm -rf model/
path/to/spark/bin/spark-submit --master local[2] --executor-memory 4g __init__.py