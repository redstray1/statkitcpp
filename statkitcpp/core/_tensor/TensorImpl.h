#pragma once
#include <stdexcept>
#include <vector>
#include "Storage.h"
#include "datatypes.h"
#include "ScalarType.h"
#include "../_autograd/autograd.h"
#include <memory>
#include "Node.h"

namespace statkitcpp {

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
friend class Tensor;
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    Storage storage_;
    ScalarType dtype_;
    size_t size_;
    size_t storage_offset_;

    bool requires_grad_ = false;

    size_t GetFlatIndex(const std::vector<size_t>& indexes) const;
    std::vector<size_t> GetIndexesFromFlat(size_t flat_index) const;
    
    void RecursiveToString(size_t depth, size_t& cur_index, std::string& result) const;

    template <typename T>
    inline const T* ConstDataPtr() const {
        return GetDataPtrImpl<const T>(
            [this] { return static_cast<const T*>(storage_.GetDataPtr());});
    }

    template <typename T>
    inline T* TDataPtr() {
        return GetDataPtrImpl<T>(
            [this] { return static_cast<T*>(storage_.GetDataPtr());});
    }

    template <typename T, typename Func>
    T* GetDataPtrImpl(const Func& get_data) const {
        if (!storage_) {
            throw std::runtime_error{"Can not get pointer to uninitialized storage"};
        }
        return get_data() + storage_offset_;
    }
    std::shared_ptr<Node> autograd_node_ = nullptr;
public:
    std::shared_ptr<GradFunction> grad_fn = nullptr;
    std::shared_ptr<Tensor> grad = nullptr;

    TensorImpl();
    explicit TensorImpl(const std::vector<size_t>& shape,
                        ScalarType dtype = kFloat32,
                        bool requires_grad = false);
    explicit TensorImpl(void* data, 
                        const std::vector<size_t>& shape,
                        ScalarType dtype = kFloat32,
                        bool requires_grad = false);
    explicit TensorImpl(const Storage& storage,
                        const std::vector<size_t>& shape,
                        ScalarType dtype = kFloat32,
                        bool requires_grad = false);
    explicit TensorImpl(const Scalar& scalar,
                        ScalarType dtype = kFloat32,
                        bool requires_grad = false);
    TensorImpl(const TensorImpl& other) = delete;
    TensorImpl(TensorImpl&& other) = delete;
    ~TensorImpl() {}

    TensorImpl& operator=(const TensorImpl& other) = delete;
    TensorImpl& operator=(TensorImpl&& other) = delete;

    ScalarType GetDType() const { return dtype_; }
    std::string GetTypeName() const;
    std::string ToString() const;

    const std::vector<size_t> GetShape() const;
    void SetShape(const std::vector<size_t>& shape);
    void Reshape(const std::vector<size_t>& new_shape);
    void Unsqueeze(int dim);

    const std::vector<size_t> GetStrides() const;
    size_t GetSize() const;
    size_t GetNDim() const;
    size_t GetDim(int dim) const;
    size_t GetStride(int dim) const;

    Tensor& GetGrad();

    void SetRequiresGrad(bool requires_grad) { requires_grad_ = requires_grad; };
    bool GetRequiresGrad() const { return requires_grad_; };
    bool IsLeaf() const { return grad_fn == nullptr; }

    void*& GetDataPointer()  { return storage_.GetDataPtr(); }
    Storage& GetStorage() { return storage_; }
    size_t GetItemSize() const  { return ItemSize(dtype_); }
    size_t GetNBytes() const  { return GetSize() * GetItemSize(); }

    void AddGrad(const Tensor& to_add);

    std::shared_ptr<Node>& GetAutogradNode();

};

}