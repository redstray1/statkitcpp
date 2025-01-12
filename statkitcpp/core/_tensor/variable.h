#ifndef VARIABLE_HEADER_H
#define VARIABLE_HEADER_H
#include <string>
#include <vector>
#include <cstdint>

class Variable {
public:
    virtual ~Variable() {};
    virtual std::string ToString() const = 0;
    virtual std::vector<uint32_t> GetShape() const = 0;
    virtual void SetShape(const std::vector<uint32_t>& shape) = 0;
    virtual void Reshape(const std::vector<uint32_t>& new_shape) = 0;

    virtual std::vector<uint32_t> GetStrides() const = 0;

    virtual uint32_t GetSize() const = 0;
    
    virtual uint32_t GetNDim() const = 0;

    virtual void SetRequiresGrad(bool requires_grad) = 0;
    virtual bool GetRequiresGrad() const = 0;

    virtual void* GetDataPointer() = 0;
    virtual uint32_t GetItemSize() const = 0;
    virtual uint32_t GetNBytes() const = 0;
};
#endif