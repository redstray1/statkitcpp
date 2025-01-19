#ifndef TENSOR_HEADER_H
#define TENSOR_HEADER_H
#include <vector>
#include <math.h>
#include <sys/types.h>
#include <memory>
#include <string>
//#include "tensor_dispatcher.h"
#include "../datatypes.h"
#include "Scalar.h"
#include "ScalarType.h"
#include "TensorImpl.h"
#include "../_autograd/autograd.h"
#include "Storage.h"

namespace statkitcpp {

class Tensor {
friend class TensorDispatcher;
friend class TensorImpl;
private:
    std::shared_ptr<TensorImpl> impl_;
    bool requires_grad_ = false;

    std::string GetTypeName() const;
    size_t GetFlatIndex(const std::vector<size_t>& indexes) const {
        return impl_->GetFlatIndex(indexes);
    }
    std::vector<size_t> GetIndexesFromFlat(size_t flat_index) const {
        return impl_->GetIndexesFromFlat(flat_index);
    }
    
    template <typename BinaryOperation>
    friend Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op);
public:
    std::shared_ptr<GradFunction> grad_fn = nullptr;

    Tensor();
    explicit Tensor(const std::vector<size_t>& shape,
                    ScalarType dtype = kFloat32,
                    bool requires_grad = false);
    explicit Tensor(void* data, 
                    const std::vector<size_t>& shape,
                    ScalarType dtype = kFloat32,
                    bool requires_grad = false);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other);
    ~Tensor() {}

    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    Tensor& operator=(const Scalar& other);
    Tensor& operator=(Scalar&& other);

    Tensor ToType(ScalarType t) const;

    std::string ToString() const { return impl_->ToString(); }

    std::vector<size_t> GetShape() const { return impl_->GetShape(); };
    void SetShape(const std::vector<size_t>& shape) { impl_->SetShape(shape); }
    void Reshape(const std::vector<size_t>& new_shape) {impl_->Reshape(new_shape); };

    std::vector<size_t> GetStrides() const { return impl_->GetStrides(); };

    ScalarType GetDType() const { return impl_->GetDType(); }
    size_t GetSize() const { return impl_->GetSize(); };
    size_t GetNDim() const { return impl_->GetNDim(); };

    void SetRequiresGrad(bool requires_grad) { requires_grad_ = requires_grad; };
    bool GetRequiresGrad() const { return requires_grad_; };

    void* GetDataPointer()  { return impl_->GetDataPointer(); }
    void* GetDataPointer() const  { return impl_->GetDataPointer(); }
    Storage& GetStorage() { return impl_->GetStorage(); }
    const Storage& GetStorage() const { return impl_->GetStorage(); }
    size_t GetItemSize() const  { return impl_->GetItemSize(); }
    size_t GetNBytes() const  { return impl_->GetNBytes(); }

    bool BroadcastableTo(const Tensor& other);

    Tensor Sum(int dim = -1, bool keepdims = false) const;
    Tensor Mean(int dim = -1, bool keepdims = false) const;
    Tensor Var(int dim = -1, bool keepdims = false) const;

    Tensor Add(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor Add(const Scalar& other) const;

    Tensor Sub(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor Sub(const Scalar& other) const;

    Tensor Mul(const Tensor& other) const;
    Tensor Mul(const Scalar& other) const;

    Tensor Div(const Tensor& other) const;
    Tensor Div(const Scalar& other) const;

    // Tensor<T>& operator+=(const Tensor<T>& rhs);
    // Tensor<T> operator+(const Tensor<T>& rhs);
};

}
#endif