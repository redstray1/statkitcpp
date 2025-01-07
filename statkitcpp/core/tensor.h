#ifndef TENSOR_HEADER_H
#define TENSOR_HEADER_H
#include <cstdint>
#include <vector>
#include <math.h>
#include <sys/types.h>
#include <memory>
#include "config.h"
#include <string>

namespace statkitcpp {

template <class T = float>
class Tensor {
private:
    std::vector<uint32_t> shape_;
    data_type dtype_ = data_type::Float32;
    std::vector<T> data_;
    uint32_t size_;
    bool requires_grad_ = true;

    uint32_t GetFlatIndex(const std::vector<uint32_t>& indexes) const;
    std::vector<uint32_t> GetIndexesFromFlat(uint32_t flat_index) const;
    template <class BinaryOperation>
    static Tensor<T> ApplyBroadcastOp(const Tensor& lhs, const Tensor& rhs,
                               BinaryOperation op);
    std::string ShapeToString() const;

public:
    std::unique_ptr<Tensor> grad;

    Tensor();
    explicit Tensor(const std::vector<uint32_t>& shape);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;
    ~Tensor() {}

    static Tensor<T> Full(const std::vector<uint32_t>& shape, T value);
    static Tensor<T> Zeros(const std::vector<uint32_t>& shape);
    static Tensor<T> Ones(const std::vector<uint32_t>& shape);

    Tensor<T>& operator=(const Tensor& other) = default;
    Tensor<T>& operator=(Tensor&& other) = default;

    std::string ToString() const;
    bool BroadcastableTo(const Tensor& other);

    std::vector<uint32_t> GetShape() const;
    void SetShape(const std::vector<uint32_t>& shape);
    uint32_t GetSize() const;
    
    void SetRequiresGrad(bool requires_grad);
    bool GetRequiresGrad() const;

    std::vector<T> GetData() { return data_; }

    // Tensor<T>& operator+=(const Tensor<T>& rhs);
    // Tensor<T> operator+(const Tensor<T>& rhs);
};

}
#endif