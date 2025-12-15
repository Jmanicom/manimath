#pragma once

#include <cmath>
#include <cassert>
#include <iostream>

/// @brief 2D Vector template with various linear algebra operations
/// @tparam T Numeric type (int, unsigned int, float, double)
/// 
/// Provides a complete set of 2D vector operations including:
/// - Arithmetic operations (+, -, *, /)
/// - Vector operations (dot product, magnitude, normalization)
/// - Geometric operations (perpendicular, projection, reflection)
/// 
/// @note For floating-point types, use almostEqual() instead of ==
/// @warning Division and inverse operations assert on zero divisors
template <typename T>
struct Vec2
{
    // Check type T is of arithmetic type
    static_assert(std::is_arithmetic<T>::value, "Vec2 requires arithmetic type");

    // Components
    T x, y;

    // Constructors
    constexpr Vec2() noexcept : x(0), y(0) {}
    constexpr Vec2(T x, T y) noexcept : x(x), y(y) {}
    
    // Conversion constructor
    template <typename U>
    explicit constexpr Vec2(const Vec2<U>& other) noexcept
        : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)) {}

    // Arithmetic operators
    constexpr Vec2 operator+(const Vec2& other) const noexcept {
        return Vec2(x + other.x, y + other.y);
    }
    constexpr Vec2 operator-(const Vec2& other) const noexcept {
        return Vec2(x - other.x, y - other.y);
    }
    constexpr Vec2 operator*(T scalar) const noexcept {
        return Vec2(x * scalar, y * scalar);
    }
    friend constexpr Vec2 operator*(T scalar, const Vec2& v) noexcept {
        return v * scalar;
    }
    constexpr Vec2 operator/(T scalar) const {
        assert(scalar != T(0) && "Vec2: Division by zero");
        return Vec2(x / scalar, y / scalar);
    }
    constexpr Vec2 operator-() const noexcept {
        return Vec2(-x, -y);
    }

    // Component-wise multiplication and division
    constexpr Vec2 operator*(const Vec2& other) const noexcept {
        return Vec2(x * other.x, y * other.y);
    }
    constexpr Vec2 operator/(const Vec2& other) const {
        assert(other.x != T(0) && other.y != T(0) && "Vec2: Division by zero");
        return Vec2(x / other.x, y / other.y);
    }

    // Comparison operators
    constexpr bool operator==(const Vec2& other) const noexcept {
        return x == other.x && y == other.y;
    }
    constexpr bool operator!=(const Vec2& other) const noexcept {
        return !(*this == other);
    }

    // Epsilon-based comparison for floating-point types
    static constexpr T epsilon() noexcept {
        return std::is_floating_point<T>::value ? T(1e-6) : T(0);
    }
    constexpr bool almostEqual(const Vec2& other, T eps = epsilon()) const noexcept {
        return std::abs(x - other.x) <= eps && std::abs(y - other.y) <= eps;
    }

    // Compound assignment operators
    constexpr Vec2& operator+=(const Vec2& other) noexcept {
        x += other.x;
        y += other.y;
        return *this;
    }
    constexpr Vec2& operator-=(const Vec2& other) noexcept {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    constexpr Vec2& operator*=(T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    constexpr Vec2& operator/=(T scalar) {
        assert(scalar != T(0) && "Vec2: Division by zero");
        x /= scalar;
        y /= scalar;
        return *this;
    }

    // Vector operations
    constexpr T dot(const Vec2& other) const noexcept {
        return x * other.x + y * other.y;
    }
    
    T magnitude() const noexcept {
        return std::sqrt(x * x + y * y);
    }

    Vec2 normalized() const noexcept {
        T mag = magnitude();
        return mag > T(0) ? (*this / mag) : Vec2();
    }
    
    Vec2& normalize() noexcept {
        T mag = magnitude();
        if (mag > T(0)) {
            x /= mag;
            y /= mag;
        }
        return *this;
    }
    
    constexpr Vec2 perp() const noexcept {
        return Vec2(-y, x);
    }

    // Geometric operations
    T distance(const Vec2& other) const noexcept {
        return (*this - other).magnitude();
    }
};
// Type aliases
using Vec2i = Vec2<int>;
using Vec2u = Vec2<unsigned int>;
using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;


/// @brief 3D Vector template with various linear algebra operations
/// @tparam T Numeric type (int, unsigned int, float, double)
template <typename T>
struct Vec3
{
    static_assert(std::is_arithmetic<T>::value, "Vec3 requires arithmetic type");

    // Components
    T x, y, z;

    // Constructors
    constexpr Vec3() noexcept : x(0), y(0), z(0) {}
    constexpr Vec3(T x, T y, T z) noexcept : x(x), y(y), z(z) {}
    
    // Conversion constructor
    template <typename U>
    explicit constexpr Vec3(const Vec3<U>& other) noexcept
        : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)), z(static_cast<T>(other.z)) {}

    // Arithmetic operators
    constexpr Vec3 operator+(const Vec3& other) const noexcept {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    constexpr Vec3 operator-(const Vec3& other) const noexcept {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    constexpr Vec3 operator*(T scalar) const noexcept {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    friend constexpr Vec3 operator*(T scalar, const Vec3& v) noexcept {
        return v * scalar;
    }
    constexpr Vec3 operator/(T scalar) const {
        assert(scalar != T(0) && "Vec3: Division by zero");
        return Vec3(x / scalar, y / scalar, z / scalar);
    }
    constexpr Vec3 operator-() const noexcept {
        return Vec3(-x, -y, -z);
    }

    // Component-wise multiplication and division
    constexpr Vec3 operator*(const Vec3& other) const noexcept {
        return Vec3(x * other.x, y * other.y, z * other.z);
    }
    constexpr Vec3 operator/(const Vec3& other) const {
        assert(other.x != T(0) && other.y != T(0) && other.z != T(0) && "Vec3: Division by zero");
        return Vec3(x / other.x, y / other.y, z / other.z);
    }

    // Comparison operators
    constexpr bool operator==(const Vec3& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
    constexpr bool operator!=(const Vec3& other) const noexcept {
        return !(*this == other);
    }

    // Epsilon-based comparison
    static constexpr T epsilon() noexcept {
        return std::is_floating_point<T>::value ? T(1e-6) : T(0);
    }
    constexpr bool almostEqual(const Vec3& other, T eps = epsilon()) const noexcept {
        return std::abs(x - other.x) <= eps && 
               std::abs(y - other.y) <= eps && 
               std::abs(z - other.z) <= eps;
    }

    // Compound assignment operators
    constexpr Vec3& operator+=(const Vec3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    constexpr Vec3& operator-=(const Vec3& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    constexpr Vec3& operator*=(T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    constexpr Vec3& operator/=(T scalar) {
        assert(scalar != T(0) && "Vec3: Division by zero");
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    // Vector operations
    constexpr T dot(const Vec3& other) const noexcept {
        return x * other.x + y * other.y + z * other.z;
    }
    
    constexpr Vec3 cross(const Vec3& other) const noexcept {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    T magnitude() const noexcept {
        return std::sqrt(x * x + y * y + z * z);
    }
     
    Vec3 normalized() const noexcept {
        T mag = magnitude();
        return mag > T(0) ? (*this / mag) : Vec3();
    }
    
    Vec3& normalize() noexcept {
        T mag = magnitude();
        if (mag > T(0)) {
            x /= mag;
            y /= mag;
            z /= mag;
        }
        return *this;
    }

    // Geometric operations
    T distance(const Vec3& other) const noexcept {
        return (*this - other).magnitude();
    }
};
// Type aliases
using Vec3i = Vec3<int>;
using Vec3u = Vec3<unsigned int>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;

/// @brief 4D Vector template with various linear algebra operations
/// @tparam T Numeric type (int, unsigned int, float, double)
template <typename T>
struct Vec4
{
    static_assert(std::is_arithmetic<T>::value, "Vec4 requires arithmetic type");

    // Components
    T x, y, z, w;

    // Constructors
    constexpr Vec4() noexcept : x(0), y(0), z(0), w(0) {}
    constexpr Vec4(T x, T y, T z, T w) noexcept : x(x), y(y), z(z), w(w) {}
    
    // Conversion constructor
    template <typename U>
    explicit constexpr Vec4(const Vec4<U>& other) noexcept
        : x(static_cast<T>(other.x)), y(static_cast<T>(other.y)), 
          z(static_cast<T>(other.z)), w(static_cast<T>(other.w)) {}

    // Arithmetic operators
    constexpr Vec4 operator+(const Vec4& other) const noexcept {
        return Vec4(x + other.x, y + other.y, z + other.z, w + other.w);
    }
    constexpr Vec4 operator-(const Vec4& other) const noexcept {
        return Vec4(x - other.x, y - other.y, z - other.z, w - other.w);
    }
    constexpr Vec4 operator*(T scalar) const noexcept {
        return Vec4(x * scalar, y * scalar, z * scalar, w * scalar);
    }
    friend constexpr Vec4 operator*(T scalar, const Vec4& v) noexcept {
        return v * scalar;
    }
    constexpr Vec4 operator/(T scalar) const {
        assert(scalar != T(0) && "Vec4: Division by zero");
        return Vec4(x / scalar, y / scalar, z / scalar, w / scalar);
    }
    constexpr Vec4 operator-() const noexcept {
        return Vec4(-x, -y, -z, -w);
    }

    // Component-wise multiplication and division
    constexpr Vec4 operator*(const Vec4& other) const noexcept {
        return Vec4(x * other.x, y * other.y, z * other.z, w * other.w);
    }
    constexpr Vec4 operator/(const Vec4& other) const {
        assert(other.x != T(0) && other.y != T(0) && other.z != T(0) && other.w != T(0) && 
               "Vec4: Division by zero");
        return Vec4(x / other.x, y / other.y, z / other.z, w / other.w);
    }

    // Comparison operators
    constexpr bool operator==(const Vec4& other) const noexcept {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }
    constexpr bool operator!=(const Vec4& other) const noexcept {
        return !(*this == other);
    }

    // Epsilon-based comparison
    static constexpr T epsilon() noexcept {
        return std::is_floating_point<T>::value ? T(1e-6) : T(0);
    }
    constexpr bool almostEqual(const Vec4& other, T eps = epsilon()) const noexcept {
        return std::abs(x - other.x) <= eps && 
               std::abs(y - other.y) <= eps && 
               std::abs(z - other.z) <= eps && 
               std::abs(w - other.w) <= eps;
    }

    // Compound assignment operators
    constexpr Vec4& operator+=(const Vec4& other) noexcept {
        x += other.x; y += other.y; z += other.z; w += other.w;
        return *this;
    }
    constexpr Vec4& operator-=(const Vec4& other) noexcept {
        x -= other.x; y -= other.y; z -= other.z; w -= other.w;
        return *this;
    }
    constexpr Vec4& operator*=(T scalar) noexcept {
        x *= scalar; y *= scalar; z *= scalar; w *= scalar;
        return *this;
    }
    constexpr Vec4& operator/=(T scalar) {
        assert(scalar != T(0) && "Vec4: Division by zero");
        x /= scalar; y /= scalar; z /= scalar; w /= scalar;
        return *this;
    }

    // Vector operations
    constexpr T dot(const Vec4& other) const noexcept {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }
    
    T magnitude() const noexcept {
        return std::sqrt(x * x + y * y + z * z + w * w);
    }
    
    Vec4 normalized() const noexcept {
        T mag = magnitude();
        return mag > T(0) ? (*this / mag) : Vec4();
    }
    
    Vec4& normalize() noexcept {
        T mag = magnitude();
        if (mag > T(0)) {
            x /= mag;
            y /= mag;
            z /= mag;
            w /= mag;
        }
        return *this;
    }

    // Geometric operations
    T distance(const Vec4& other) const noexcept {
        return (*this - other).magnitude();
    }
};
// Type aliases
using Vec4i = Vec4<int>;
using Vec4u = Vec4<unsigned int>;
using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;

/// @brief A 2x2 matrix template with basic linear algebra operations
/// @tparam T Numeric type (int, unsigned int, float, double)
template <typename T>
struct Mat2
{
    static_assert(std::is_arithmetic<T>::value, "Mat2 requires arithmetic type");

    // Matrix data in row-major order
    T m[2][2];

    // Constructors
    constexpr Mat2() noexcept {
        m[0][0] = 1; m[0][1] = 0;
        m[1][0] = 0; m[1][1] = 1;
    }
    
    constexpr Mat2(T m00, T m01, T m10, T m11) noexcept {
        m[0][0] = m00; m[0][1] = m01;
        m[1][0] = m10; m[1][1] = m11;
    }

    // Static factory methods
    static constexpr Mat2 identity() noexcept {
        return Mat2();
    }
    
    static Mat2 rotation(T angle) noexcept {
        T c = std::cos(angle);
        T s = std::sin(angle);
        return Mat2(c, -s, s, c);
    }
    
    static constexpr Mat2 scale(T sx, T sy) noexcept {
        return Mat2(sx, 0, 0, sy);
    }
    
    static constexpr Mat2 scale(T s) noexcept {
        return scale(s, s);
    }

    // Matrix-Vector multiplication
    constexpr Vec2<T> operator*(const Vec2<T>& v) const noexcept {
        return Vec2<T>(
            m[0][0] * v.x + m[0][1] * v.y,
            m[1][0] * v.x + m[1][1] * v.y
        );
    }

    // Matrix-Matrix multiplication
    constexpr Mat2 operator*(const Mat2& other) const noexcept {
        return Mat2(
            m[0][0] * other.m[0][0] + m[0][1] * other.m[1][0],
            m[0][0] * other.m[0][1] + m[0][1] * other.m[1][1],
            m[1][0] * other.m[0][0] + m[1][1] * other.m[1][0],
            m[1][0] * other.m[0][1] + m[1][1] * other.m[1][1]
        );
    }
    
    // Matrix operations
    constexpr Mat2 transposed() const noexcept {
        return Mat2(m[0][0], m[1][0], m[0][1], m[1][1]);
    }

    constexpr T det() const noexcept {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    Mat2 inverse() const {
        T determinant = det();
        assert(std::abs(determinant) > std::numeric_limits<T>::epsilon() && 
               "Mat2: Matrix is singular (determinant is zero)");
        T invdet = T(1) / determinant;
        return Mat2(
            m[1][1] * invdet, -m[0][1] * invdet,
            -m[1][0] * invdet, m[0][0] * invdet
        );
    }

    // Element access
    constexpr T& operator()(size_t i, size_t j) noexcept {
        assert(i < 2 && j < 2 && "Mat2: Index out of bounds");
        return m[i][j];
    }
    constexpr const T& operator()(size_t i, size_t j) const noexcept {
        assert(i < 2 && j < 2 && "Mat2: Index out of bounds");
        return m[i][j];
    }

    // Row and column access
    constexpr Vec2<T> row(size_t i) const noexcept {
        assert(i < 2 && "Mat2: Row index out of bounds");
        return Vec2<T>(m[i][0], m[i][1]);
    }
    constexpr Vec2<T> column(size_t j) const noexcept {
        assert(j < 2 && "Mat2: Column index out of bounds");
        return Vec2<T>(m[0][j], m[1][j]);
    }
};
// Type aliases
using Mat2i = Mat2<int>;
using Mat2u = Mat2<unsigned int>;
using Mat2f = Mat2<float>;
using Mat2d = Mat2<double>;

/// @brief A 3x3 matrix template with basic linear algebra operations
/// @tparam T Numeric type (int, unsigned int, float, double)
template <typename T>
struct Mat3
{
    static_assert(std::is_arithmetic<T>::value, "Mat3 requires arithmetic type");

    // Matrix data in row-major order
    T m[3][3];

    // Constructors
    constexpr Mat3() noexcept {
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0;
        m[2][0] = 0; m[2][1] = 0; m[2][2] = 1;
    }

    constexpr Mat3(T m00, T m01, T m02,
                   T m10, T m11, T m12,
                   T m20, T m21, T m22) noexcept {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }

    // Static factory methods
    static constexpr Mat3 identity() noexcept {
        return Mat3();
    }

    static Mat3 rotation(T angle) noexcept {
        T c = std::cos(angle);
        T s = std::sin(angle);
        return Mat3(
            c, -s, 0,
            s,  c, 0,
            0,  0, 1
        );
    }
    
    static Mat3 rotationX(T angle) noexcept {
        T c = std::cos(angle);
        T s = std::sin(angle);
        return Mat3(
            1, 0,  0,
            0, c, -s,
            0, s,  c
        );
    }
    
    static Mat3 rotationY(T angle) noexcept {
        T c = std::cos(angle);
        T s = std::sin(angle);
        return Mat3(
             c, 0, s,
             0, 1, 0,
            -s, 0, c
        );
    }
    
    static Mat3 rotationZ(T angle) noexcept {
        return rotation(angle);
    }

    static constexpr Mat3 scale(T sx, T sy) noexcept {
        return Mat3(
            sx, 0, 0,
            0, sy, 0,
            0,  0, 1
        );
    }
    
    static constexpr Mat3 scale(T s) noexcept {
        return scale(s, s);
    }

    static constexpr Mat3 translation(T tx, T ty) noexcept {
        return Mat3(
            1, 0, tx,
            0, 1, ty,
            0, 0, 1
        );
    }

    // Matrix-Vector multiplication (homogeneous coordinates)
    constexpr Vec3<T> operator*(const Vec3<T>& v) const noexcept {
        return Vec3<T>(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }

    // Transform a 2D point (assumes z=1 for translation)
    constexpr Vec2<T> transformPoint(const Vec2<T>& v) const noexcept {
        return Vec2<T>(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2],
            m[1][0] * v.x + m[1][1] * v.y + m[1][2]
        );
    }

    // Transform a 2D vector (ignores translation, assumes z=0)
    constexpr Vec2<T> transformVector(const Vec2<T>& v) const noexcept {
        return Vec2<T>(
            m[0][0] * v.x + m[0][1] * v.y,
            m[1][0] * v.x + m[1][1] * v.y
        );
    }

    // Matrix-Matrix multiplication
    constexpr Mat3 operator*(const Mat3& other) const noexcept {
        Mat3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }

    // Matrix operations
    constexpr Mat3 transposed() const noexcept {
        return Mat3(
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2]
        );
    }

    constexpr T det() const noexcept {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    Mat3 inverse() const {
        T determinant = det();
        assert(std::abs(determinant) > std::numeric_limits<T>::epsilon() && 
               "Mat3: Matrix is singular (determinant is zero)");
        T invDet = T(1) / determinant;
    
        return Mat3(
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet,
        
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet,
        
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet
        );
    }

    // Element access
    constexpr T& operator()(size_t i, size_t j) noexcept {
        assert(i < 3 && j < 3 && "Mat3: Index out of bounds");
        return m[i][j];
    }
    constexpr const T& operator()(size_t i, size_t j) const noexcept {
        assert(i < 3 && j < 3 && "Mat3: Index out of bounds");
        return m[i][j];
    }

    // Row and column access
    constexpr Vec3<T> row(size_t i) const noexcept {
        assert(i < 3 && "Mat3: Row index out of bounds");
        return Vec3<T>(m[i][0], m[i][1], m[i][2]);
    }
    constexpr Vec3<T> column(size_t j) const noexcept {
        assert(j < 3 && "Mat3: Column index out of bounds");
        return Vec3<T>(m[0][j], m[1][j], m[2][j]);
    }
};
// Type aliases
using Mat3i = Mat3<int>;
using Mat3u = Mat3<unsigned int>;
using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;

// Hash support for unordered containers
namespace std {
    template <typename T>
    struct hash<Vec2<T>> {
        size_t operator()(const Vec2<T>& v) const noexcept {
            size_t h1 = hash<T>{}(v.x);
            size_t h2 = hash<T>{}(v.y);
            return h1 ^ (h2 << 1);
        }
    };
    
    template <typename T>
    struct hash<Vec3<T>> {
        size_t operator()(const Vec3<T>& v) const noexcept {
            size_t h1 = hash<T>{}(v.x);
            size_t h2 = hash<T>{}(v.y);
            size_t h3 = hash<T>{}(v.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    
    template <typename T>
    struct hash<Vec4<T>> {
        size_t operator()(const Vec4<T>& v) const noexcept {
            size_t h1 = hash<T>{}(v.x);
            size_t h2 = hash<T>{}(v.y);
            size_t h3 = hash<T>{}(v.z);
            size_t h4 = hash<T>{}(v.w);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };
}