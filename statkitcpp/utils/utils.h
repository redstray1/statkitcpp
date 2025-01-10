#include <string>
#include <sstream>

#define STRING(Value) #Value 

template <typename T>
std::string FloatRepr(T a) {
    std::stringstream ss;
    ss << a;
    return ss.str();
}