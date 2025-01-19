#pragma once

#include "Scalar.h"
#include "Tensor.h"

namespace statkitcpp {
    namespace ops {
        void full(Tensor& data, const Scalar& fill_value); //NOLINT

        void zeros(Tensor& data); //NOLINT

        void ones(Tensor& data); //NOLINT

        void arange(Tensor& data, const Scalar& start, const Scalar& end, const Scalar& step); //NOLINT

    }
}