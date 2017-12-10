#define EIGEN_USE_THREADS

#include <cmath>
#include <climits>
#include <algorithm>
#include <string>
#include <unordered_set>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

using namespace std;
using namespace tensorflow;

REGISTER_OP("StringFiltering")
  .Input("input_string: string")
  .Output("output_string: string")
  .Attr("loewr_case: bool = False")
  .Attr("include_charset: string");


template <typename T>
class StringFiltering : public OpKernel {
public:
  explicit StringFilteringOp(OpKernelConstruction* context): OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("loewr_case", &loewr_case_));
    string charset_string;
    OP_REQUIRES_OK(context,
                   context->GetAttr("include_charset", &charset_string));
    for (char c : charset_string) {
      charset_.insert(c);
    }
  }

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
      string processed_string = "";
      if (loewr_case_) {
        transform(orig_string.begin(), orig_string.end(), orig_string.begin(), ::tolower);
      }
      for (char c : orig_string) {
        if (charset_.find(c) != charset_.end()) {
          processed_string += c;
        }
      }
      output_string_tensor(i) = processed_string;
    }
  }

private:
  unordered_set<char> charset_;
  bool loewr_case_;
};

REGISTER_KERNEL_BUILDER(Name("StringFiltering").Device(DEVICE_CPU),
                        StringFilteringOp)
