#ifndef TENSOR_HEADER_H
#define TENSOR_HEADER_H
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <math.h>
#include <sys/types.h>
#include <memory>
#include <string>
#include "variable.h"
#include "../_autograd/autograd.h"

namespace statkitcpp {

template <class T = float>
class Tensor : public Variable {
private:
    std::vector<uint32_t> shape_;
    std::vector<uint32_t> strides_;
    std::vector<T> data_;
    uint32_t size_;
    bool requires_grad_ = false;
    std::string GetTypeName() const;
    uint32_t GetFlatIndex(const std::vector<uint32_t>& indexes) const;
    std::vector<uint32_t> GetIndexesFromFlat(uint32_t flat_index) const;
    template <class BinaryOperation>
    static Tensor<T> ApplyBroadcastOp(const Tensor& lhs, const Tensor& rhs,
                               BinaryOperation op);
    void RecursiveToString(uint32_t depth, uint32_t& cur_index, std::string& result) const;
public:
    std::shared_ptr<GradFunction> grad_fn;

    Tensor();
    explicit Tensor(const std::vector<uint32_t>& shape, bool requires_grad = false);
    explicit Tensor(T* data, 
                    const std::vector<uint32_t>& shape,
                    bool requires_grad = false);
    explicit Tensor(std::initializer_list<T> data,
                    const std::vector<uint32_t>& shape,
                    bool requires_grad = false);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other);
    ~Tensor() {}

    static Tensor<T> Full(const std::vector<uint32_t>& shape, const T& value);
    static Tensor<T> Zeros(const std::vector<uint32_t>& shape);
    static Tensor<T> Ones(const std::vector<uint32_t>& shape);

    Tensor<T>& operator=(const Tensor& other) = default;
    Tensor<T>& operator=(Tensor&& other) = default;

    template <typename U, typename = typename std::enable_if<std::is_convertible<T, U>::value>::type>
    operator Tensor<U>() {
        Tensor<U> output(shape_, requires_grad_);
        for (size_t i = 0; i < size_; i++) {
            output.GetData()[i] = static_cast<U>(data_[i]);
        }
        return output;
    }

    std::string ToString() const override;
    bool BroadcastableTo(const Tensor& other);

    std::vector<uint32_t> GetShape() const override;
    void SetShape(const std::vector<uint32_t>& shape) override;
    void Reshape(const std::vector<uint32_t>& new_shape) override;

    std::vector<uint32_t> GetStrides() const override;

    uint32_t GetSize() const override;
    
    uint32_t GetNDim() const override;

    void SetRequiresGrad(bool requires_grad) override;
    bool GetRequiresGrad() const override;

    void* GetDataPointer() override { return data_.data(); }
    uint32_t GetItemSize() const override { return sizeof(T); }
    uint32_t GetNBytes() const override { return GetSize() * GetItemSize(); }

    // Tensor<T>& operator+=(const Tensor<T>& rhs);
    // Tensor<T> operator+(const Tensor<T>& rhs);
};

}
#endif