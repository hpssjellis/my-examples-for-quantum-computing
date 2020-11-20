#!/bin/bash


echo "Enter file name of python file showing errors"
echo "making an errors.txt file to append these errors"
echo ""
read wow4

#mkdir $wow4

(echo ""; echo "*********************"; echo "*********************"; echo "Running Python file: $wow4"; echo "*********************"; echo ""; ) >> /workspace/my-examples-for-quantum-computing/pennylaneai/qml-demos/errors.txt

python3 $wow4.py 2>> /workspace/my-examples-for-quantum-computing/pennylaneai/qml-demos/errors.txt
