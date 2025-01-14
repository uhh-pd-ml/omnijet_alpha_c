#!/bin/bash
curl --output https://syncandshare.desy.de/index.php/s/PDJmkdBodD9TYRB/download\?path\=\&files\=checkpoints.tar.gz
tar -xvf checkpoints.tar.gz
rm -rf checkpoints.tar.gz
