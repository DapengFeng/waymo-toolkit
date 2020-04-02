#!/bin/bash

protoc -I=. --python_out=. waymo_toolkit/protos/label.proto
protoc -I=. --python_out=. waymo_toolkit/protos/dataset.proto
protoc -I=. --python_out=. waymo_toolkit/protos/annotation.proto 