#ifndef TENSOR_HEADER_H
#define TENSOR_HEADER_H
#include <cstddef>
#include <vector>
#include <math.h>
#include <sys/types.h>
#include <memory>
#include <string>
#include <optional>
//#include "tensor_dispatcher.h"
#include "../datatypes.h"
#include "Scalar.h"
#include "ScalarType.h"
#include "TensorImpl.h"
#include "../_autograd/autograd.h"
#include "tensor_defines.h"
#include "Storage.h"

namespace statkitcpp {

class Tensor {
friend class TensorDispatcher;
friend class TensorImpl;
friend class TensorBody;
private:

    std::shared_ptr<TensorImpl> impl_;

    std::string GetTypeName() const;
    size_t GetFlatIndex(const std::vector<size_t>& indexes) const {
        return impl_->GetFlatIndex(indexes);
    }
    std::vector<size_t> GetIndexesFromFlat(size_t flat_index) const {
        return impl_->GetIndexesFromFlat(flat_index);
    }
    
    // template <typename BinaryOperation>
    // friend Tensor ApplyBinaryOp(const Tensor& lhs, const Tensor& rhs, BinaryOperation op);
public:
    // std::shared_ptr<GradFunction> grad_fn = nullptr;
    // std::shared_ptr<Tensor> grad = nullptr;

    Tensor();
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    explicit Tensor(const std::vector<size_t>& shape,
                    ScalarType dtype = kFloat32,
                    bool requires_grad = false);
    explicit Tensor(void* data, 
                    const std::vector<size_t>& shape,
                    ScalarType dtype = kFloat32,
                    bool requires_grad = false);
    explicit Tensor(const Storage& storage,
                    const std::vector<size_t>& shape,
                    ScalarType dtype = kFloat32,
                    bool requires_grad = false);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other);
    ~Tensor() {}

    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    Tensor& operator=(const Scalar& other);

    Tensor ToType(ScalarType t) const;

    std::string ToString() const;

    const std::vector<size_t> GetShape() const { return impl_->GetShape(); };
    void SetShape(const std::vector<size_t>& shape) { *this = Reshape(shape); }
    Tensor Reshape(const std::vector<size_t>& shape);

    const std::vector<size_t> GetStrides() const { return impl_->GetStrides(); };

    ScalarType GetDType() const { return impl_->GetDType(); }
    size_t GetSize() const { return impl_->GetSize(); };
    size_t GetNDim() const { return impl_->GetNDim(); };

    void SetRequiresGrad(bool requires_grad) { impl_->SetRequiresGrad(requires_grad); };
    bool GetRequiresGrad() const { return impl_->GetRequiresGrad(); };
    bool IsLeaf() const { return impl_->IsLeaf(); }

    void* GetDataPointer()  { return impl_->GetDataPointer(); }
    void* GetDataPointer() const  { return impl_->GetDataPointer(); }
    Storage& GetStorage() { return impl_->GetStorage(); }
    const Storage& GetStorage() const { return impl_->GetStorage(); }
    size_t GetItemSize() const  { return impl_->GetItemSize(); }
    size_t GetNBytes() const  { return impl_->GetNBytes(); }

    Tensor& GetGrad() const { return impl_->GetGrad(); }
    std::shared_ptr<TensorImpl> GetImpl() const { return impl_; }


    bool BroadcastableTo(const Tensor& other);

    void Backward(std::optional<Tensor> grad_output = std::nullopt, std::optional<Tensor> output = std::nullopt, bool retain_graph=false) &;

    TENSOR_AGGREGATION_METHODS(AGGREGATION_DECLARATIONS)

    TENSOR_BINARY_DECLARATIONS_WITH_OP(OPERATOR_METHODS_DECLARATIONS)
    TENSOR_BINARY_DECLARATIONS_WITH_OP(OPERATORS_DEFINITIONS)

    TENSOR_BINARY_DECLARATIONS_WITHOUT_OP(OPERATOR_METHODS_DECLARATIONS)

    TENSOR_POINTWISE_METHODS(POINTWISE_DECLARATIONS)
};

}
#endif