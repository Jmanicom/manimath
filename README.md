# manimath

A custom built C++ math library for 2D/3D physics simulation engines
## Modules

### [Linear Algebra](matvec.hpp) 
**`matvec.hpp`** - Vectors and matrices for use in 2D/3D physics simulations or graphics

Provides templated vector and matrix types with comprehensive operations including dot products, cross products, matrix transformations, and more.
```cpp
#include "matvec.hpp"

Vec2f position(10.0f, 5.0f);
Mat2f rotation = Mat2f::rotation(pi / 4);  // 45 degrees
Vec2f rotated = rotation * position;
```

**[View Documentation →](matvec/README.md)**

---

*More modules coming soon! Planned additions include:*
- **Geometry** - Shapes, bounding boxes, primitives
- **Collision** - Detection and response algorithms
- **Utils** - Common math utilities and constants

## Quick Start

### Installation

**Option 1: Download a single module**
```bash
# Download just what you need
curl -O https://raw.githubusercontent.com/Jmanicom/manimath/main/matvec.hpp
```

**Option 2: Clone the entire repository**
```bash
git clone https://github.com/Jmanicom/manimath.git
```

**Option 3: Add as a Git submodule**
```bash
git submodule add https://github.com/Jmanicom/manimath.git external/manimath
```

### Usage

Simply include the header file you need:
```cpp
#include "matvec.hpp"

int main() {
    // Create vectors
    Vec2f v1(3.0f, 4.0f);
    Vec2f v2(1.0f, 2.0f);
    
    // Perform operations
    Vec2f sum = v1 + v2;
    float dot = v1.dot(v2);
    float magnitude = v1.magnitude();
    
    // Matrix transformations
    Mat3f transform = Mat3f::translation(100.0f, 50.0f) 
                    * Mat3f::rotation(0.785f)
                    * Mat3f::scale(2.0f, 2.0f);
    
    Vec2f transformed = transform.transformPoint(v1);
    
    return 0;
}
```

### Compilation
```bash
# Basic compilation (C++11 or later)
g++ -std=c++11 your_program.cpp -o your_program

# With optimization
g++ -std=c++11 -O3 your_program.cpp -o your_program

# Include from custom location
g++ -std=c++11 -I/path/to/manimath/linear-algebra your_program.cpp -o your_program
```

## Examples

manimath was developed to be used with a 2D physics simulation project I had been working on: [momentum](https://github.com/Jmanicom/momentum).

The source code can be viewed to see how manimath was used in tracking various objects position, velocity, and acceleration vectors as well as some vector arithmetic for collision algoritihims.

## Features

- **Header-only** - No compilation or linking required
- **Zero dependencies** - Only uses C++ standard library
- **Templated** - Works with `int`, `unsigned int`, `float`, and more
- **Modern C++** - Uses `constexpr` for compile-time computation where possible
- **Type-safe** - Strong compile-time type checking
- **Well-documented** - Comprehensive inline documentation

## Requirements

- **C++11 or later**
- Any standard-compliant C++ compiler (GCC, Clang, MSVC, etc.)

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or request features via [Issues](https://github.com/Jmanicom/manimath/issues)
- Submit pull requests
- Improve documentation
- Add examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Each module can be used independently under the same license.

## Acknowledgments

Inspired by:
- [GLM](https://github.com/g-truc/glm) - OpenGL Mathematics
- [Eigen](https://eigen.tuxfamily.org/) - C++ template library for linear algebra
---
