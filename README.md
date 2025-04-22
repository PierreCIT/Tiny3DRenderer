# Tiny3DRenderer
Following a lesson from https://github.com/ssloy/tinyrenderer, it will implement a tiny 3D renderer


## C++ version
First step is to unzip the `obj` files. Then build and run.
```bash
unzip obj.zip
mkdir -p build
cd build && cmake ../src && make -j3
./Tiny3DRenderer
```

The code has been executed and the results are in the image folder.

## Python version
Install package:
```bash
cd ./src
pip install -r requirements.txt -e .
```

### Tests
```bash
pytest -v
```