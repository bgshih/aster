#include <algorithm>
#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

using namespace std;
using namespace tensorflow;

REGISTER_OP("StringReverse")
  .Input("input_string: string")
  .Output("output_string: string")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    using namespace shape_inference;
    ShapeHandle input_string = c->input(0);
    TF_RETURN_IF_ERROR(c->WithRank(input_string, 1, &input_string));
    DimensionHandle num_strings = c->Dim(input_string, 0);
    c->set_output(0, c->MakeShape({num_strings}));
    return Status::OK();
  });


class StringReverseOp : public OpKernel {
public:
  explicit StringReverseOp(OpKernelConstruction* context): OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // input-0 input_string
    const Tensor& input_string = context->input(0);
    OP_REQUIRES(context, input_string.dims() == 1,
                errors::InvalidArgument("Expected 1D string input, got ",
                                        input_string.shape().DebugString()));
    auto input_string_tensor = input_string.tensor<string, 1>();

    const int num_strings = input_string.dim_size(0);

    Tensor* output_string = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {num_strings}, &output_string));
    auto output_string_tensor = output_string->tensor<string, 1>();
    
    for (int i = 0; i < num_strings; i++) {
      string orig_string = input_string_tensor(i);
      string reversed_string = orig_string;
      reverse(reversed_string.begin(), reversed_string.end());
      output_string_tensor(i) = reversed_string;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StringReverse").Device(DEVICE_CPU),
                        StringReverseOp)
