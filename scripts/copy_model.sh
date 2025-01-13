#!/bin/sh

cp $1/infos_$2-best.pkl $1/infos_$3-best.pkl 
cp $1/infos_$2.pkl $1/infos_$3.pkl 
cp $1/histories_$2.pkl $1/histories_$3.pkl 
cp $1/model-$2.pth $1/model-$3.pth
