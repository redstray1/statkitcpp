#include <cstdint>
#include <vector>
#include <math.h>
#include <memory>
#include "config.h"

namespace statkitcpp {

template <class T = float>
class Tensor {
private:
    std::vector<uint32_t> shape_;
    data_type dtype_ = data_type::Float32;
    std::vector<T> data_;
    uint32_t size_;
    bool requires_grad_;

public:
    std::unique_ptr<Tensor> grad;

    Tensor();
    explicit Tensor(const std::vector<uint32_t>& shape, data_type dtype = data_type::Float32);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;
    ~Tensor() {}

    static Tensor<T> Full(const std::vector<uint32_t>& shape, T value, data_type dtype = data_type::Float32);
    static Tensor<T> Zeros(const std::vector<uint32_t>& shape, T value, data_type dtype = data_type::Float32);
    static Tensor<T> Ones(const std::vector<uint32_t>& shape, T value, data_type dtype = data_type::Float32);

    Tensor<T>& operator=(const Tensor& other) = default;
    Tensor<T>& operator=(Tensor&& other) = default;

    std::vector<uint32_t> GetShape() const;
    void SetShape(const std::vector<uint32_t>& shape);

    uint32_t GetSize() const;
    
    void SetRequiresGrad(bool requires_grad);
    const bool GetRequiresGrad() const;
};
}