#pragma once
#include <stdexcept>
#include <vector>
#include "Storage.h"
#include "ScalarType.h"

namespace statkitcpp {

class TensorImpl {
friend class Tensor;
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    Storage storage_;
    ScalarType dtype_;
    size_t size_;
    size_t storage_offset_;

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

public:

    TensorImpl();
    explicit TensorImpl(const std::vector<size_t>& shape, ScalarType dtype);
    explicit TensorImpl(void* data, 
                        const std::vector<size_t>& shape,
                        ScalarType dtype);
    explicit TensorImpl(const Storage& data,
                        const std::vector<size_t>& shape,
                        ScalarType dtype);
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

    const std::vector<size_t> GetStrides() const;

    size_t GetSize() const ;
    
    size_t GetNDim() const ;

    void SetRequiresGrad(bool requires_grad) ;
    bool GetRequiresGrad() const ;

    void* GetDataPointer()  { return storage_.GetDataPtr(); }
    Storage& GetStorage() { return storage_; }
    size_t GetItemSize() const  { return ItemSize(dtype_); }
    size_t GetNBytes() const  { return GetSize() * GetItemSize(); }

};

}