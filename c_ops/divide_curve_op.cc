#include <cmath>
#include <iostream>
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

REGISTER_OP("DivideCurve")
  .Input("curve_points: float32")
  .Output("key_points: float32")
  .Attr("num_key_points: int = 20")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    using namespace shape_inference;

    ShapeHandle curve_points = c->input(0);
    TF_RETURN_IF_ERROR(c->WithRank(curve_points, 2, &curve_points));
    DimensionHandle batch_size = c->Dim(curve_points, 0);

    int num_key_points;
    TF_RETURN_IF_ERROR(c->GetAttr("num_key_points", &num_key_points));

    c->set_output(0, c->MakeShape({batch_size, num_key_points * 2}));
    return Status::OK();
  });


template <typename T>
class DivideCurveOp : public OpKernel {
  typedef array<T, 2> point_t;

public:
  explicit DivideCurveOp(OpKernelConstruction* context): OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_key_points", &num_key_points_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& curve_points = context->input(0);
    OP_REQUIRES(context, curve_points.dims() == 2,
                errors::InvalidArgument("Expected curve_points to be 2D, got ",
                                        curve_points.shape().DebugString()));
    OP_REQUIRES(context, curve_points.dim_size(1) % 2 == 0,
                errors::InvalidArgument("Number of curve points must be even"));
    const int num_curve_points = curve_points.dim_size(1) / 2;
    const int num_curve_points_per_side = num_curve_points / 2;
    const int num_key_points_per_side = num_key_points_ / 2;
    const int batch_size = curve_points.dim_size(0);

    Tensor* key_points = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {batch_size, num_key_points_ * 2}, &key_points));
    auto key_points_tensor = key_points->tensor<T, 2>();

    auto curve_points_tensor = curve_points.tensor<T, 2>();
    for (int i = 0; i < batch_size; i++) {
      vector<point_t> upper_curve_points_vec;
      vector<point_t> lower_curve_points_vec;
      for (int j = 0; j < num_curve_points_per_side; j++) {
        upper_curve_points_vec.push_back(
          point_t({curve_points_tensor(i, 2*j),
                   curve_points_tensor(i, 2*j+1)}));
      }
      for (int j = num_curve_points_per_side; j < 2*num_curve_points_per_side; j++) {
        lower_curve_points_vec.push_back(
          point_t({curve_points_tensor(i, 2*j),
                   curve_points_tensor(i, 2*j+1)} ));
      }

      vector<point_t> key_points_vec;
      try {
        _divide_curve(upper_curve_points_vec, num_key_points_per_side, &key_points_vec);
        _divide_curve(lower_curve_points_vec, num_key_points_per_side, &key_points_vec);
      } catch (const std::runtime_error& e) {
        cout << e.what() << endl;
        key_points_vec.clear();
        for (int j = 0; j < num_key_points_; j++) {
          key_points_vec.push_back(point_t({ 0.5, 0.5 }));
        }
      }

      for (int j = 0; j < num_key_points_; j++) {
        key_points_tensor(i, 2*j) = key_points_vec[j][0];
        key_points_tensor(i, 2*j+1) = key_points_vec[j][1];
      }
    }
  }

  void _divide_curve(const vector<point_t>& curve_points,
                     const int num_key_points,
                     vector<point_t>* key_points) {
    const int n = curve_points.size();
    vector<T> distance_cumsum(n, 0);
    for (int i = 1; i < n; i++) {
      distance_cumsum[i] = distance_cumsum[i-1] + _dist(curve_points[i-1], curve_points[i]);
    }
    T total_dist = distance_cumsum[n-1];
    T segment_length = total_dist / (num_key_points - 1);

    key_points->push_back(curve_points[0]);
    for (int keypoint_idx = 1; keypoint_idx < num_key_points-1; keypoint_idx++) {
      T length = keypoint_idx * segment_length;
      bool success = false;
      for (int i = 0; i < n; i++) {
        if (length > distance_cumsum[i-1] && length <= distance_cumsum[i]) {
          T length_to_left = length - distance_cumsum[i-1];
          T length_to_right = distance_cumsum[i] - length;
          T w0 = length_to_right / (length_to_left + length_to_right);
          T w1 = 1.0 - w0;
          point_t interpolated_point({
            w0 * curve_points[i-1][0] + w1 * curve_points[i][0],
            w0 * curve_points[i-1][1] + w1 * curve_points[i][1],
          });
          key_points->push_back(interpolated_point);
          success = true;
          break;
        }
      }
      if (!success) {
        char msg[256];
        sprintf(msg, "Internal error: failed to find neighbors %f.", length);
        throw runtime_error(msg);
      }
    }
    key_points->push_back(curve_points[n-1]);
  }

  T _dist(const point_t& pt1, const point_t& pt2) {
    T xdiff = pt1[0] - pt2[0];
    T ydiff = pt1[1] - pt2[1];
    return sqrt(xdiff * xdiff + ydiff * ydiff);
  }

private:
  int num_key_points_;
};

REGISTER_KERNEL_BUILDER(Name("DivideCurve").Device(DEVICE_CPU),
                        DivideCurveOp<float>)
