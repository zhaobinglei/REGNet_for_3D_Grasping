#include "sampling.h"
#include "ball_query.h"
#include "grouping.h"
#include "interpolate.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &BallQuery, "Ball query (CUDA)");
  m.def("group_points_forward", &GroupPointsForward, "Group points forward (CUDA)");
  m.def("group_points_backward", &GroupPointsBackward, "Group points backward (CUDA)");
  m.def("farthest_point_sample", &FarthestPointSample, "Farthest point sampling (CUDA)");
  m.def("point_search", &PointSearch, "point searching in interpolate (CUDA)");
  m.def("interpolate_forward", &InterpolateForward,"Interpolate feature forward (CUDA)");
  m.def("interpolate_backward", &InterpolateBackward, "Interpolate feature backward (CUDA)");
}
