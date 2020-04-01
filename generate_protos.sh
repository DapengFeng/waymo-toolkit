#!/bin/bash

protoc -I=. --python_out=. waymo_extractor/protos/label.proto
protoc -I=. --python_out=. waymo_extractor/protos/dataset.proto
protoc -I=. --python_out=. waymo_extractor/protos/annotation.proto 