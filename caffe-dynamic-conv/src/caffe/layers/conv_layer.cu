// #include <vector>

// #include "caffe/layers/conv_layer.hpp"

// namespace caffe {

// template <typename Dtype>
// void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) {
//   const Dtype* weight = this->blobs_[0]->gpu_data();
//   for (int i = 0; i < bottom.size(); ++i) {
//     const Dtype* bottom_data = bottom[i]->gpu_data();
//     Dtype* top_data = top[i]->mutable_gpu_data();
//     for (int n = 0; n < this->num_; ++n) {
//       this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
//           top_data + n * this->top_dim_);
//       if (this->bias_term_) {
//         const Dtype* bias = this->blobs_[1]->gpu_data();
//         this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
//       }
//     }
//   }
// }

// template <typename Dtype>
// void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//   const Dtype* weight = this->blobs_[0]->gpu_data();
//   Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
//   for (int i = 0; i < top.size(); ++i) {
//     const Dtype* top_diff = top[i]->gpu_diff();
//     // Bias gradient, if necessary.
//     if (this->bias_term_ && this->param_propagate_down_[1]) {
//       Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
//       for (int n = 0; n < this->num_; ++n) {
//         this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
//       }
//     }
//     if (this->param_propagate_down_[0] || propagate_down[i]) {
//       const Dtype* bottom_data = bottom[i]->gpu_data();
//       Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
//       for (int n = 0; n < this->num_; ++n) {
//         // gradient w.r.t. weight. Note that we will accumulate diffs.
//         if (this->param_propagate_down_[0]) {
//           this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
//               top_diff + n * this->top_dim_, weight_diff);
//         }
//         // gradient w.r.t. bottom data, if necessary.
//         if (propagate_down[i]) {
//           this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
//               bottom_diff + n * this->bottom_dim_);
//         }
//       }
//     }
//   }
// }

// INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

// }  // namespace caffe

#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int bottom_size = bottom.size();
  int top_size = top.size();
  if(bottom_size - top_size == 1){
    //luojun
    const ConvolutionParameter_WeightOp op_ = this->layer_param_.convolution_param().weight_operation();
    //
    const Dtype* bottom_weight = bottom[bottom_size - 1]->gpu_data();
    Dtype* new_weight = this->new_weight_->mutable_gpu_data();
    int weight_count = this->blobs_[0]->count();
    for (int i = 0; i < top.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      for (int n = 0; n < this->num_; ++n) {
        switch(op_){ 
        case ConvolutionParameter_WeightOp_MUL:
          caffe_gpu_mul(weight_count, weight, bottom_weight + n * weight_count, new_weight);
          break;
        case ConvolutionParameter_WeightOp_ADD:
          caffe_gpu_add(weight_count, weight, bottom_weight + n * weight_count, new_weight);
          break;
        case ConvolutionParameter_WeightOp_COPY:
          caffe_gpu_memcpy(weight_count, bottom_weight + n * weight_count, new_weight);
          break;
        default:
          LOG(FATAL) << "Unknown weight operation.";
        }
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, new_weight,
            top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    }

  }
  else{
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      for (int n = 0; n < this->num_; ++n) {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  int bottom_size = bottom.size();
  int top_size = top.size();
  if (bottom_size - top_size == 1){
        //luojun
    const ConvolutionParameter_WeightOp op_ = this->layer_param_.convolution_param().weight_operation();
    //
    Dtype* bottom_weight = bottom[bottom_size - 1]->mutable_gpu_data();
    Dtype* bottom_weight_diff = bottom[bottom_size - 1]->mutable_gpu_diff();
    Dtype* new_weight = this->new_weight_->mutable_gpu_data();
    Dtype* new_weight_diff = this->new_weight_->mutable_gpu_diff();
    int weight_count = this->blobs_[0]->count();
    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();
      // Bias gradient, if necessary.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
        }
      }
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm2(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, new_weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_gpu_gemm(top_diff + n * this->top_dim_, new_weight,
                bottom_diff + n * this->bottom_dim_);
          }
          switch(op_){ 
            case ConvolutionParameter_WeightOp_MUL:
              caffe_gpu_mul(weight_count, new_weight_diff, bottom_weight + n * weight_count, new_weight);
              caffe_gpu_axpy(weight_count, Dtype(1.0), new_weight, weight_diff);
              caffe_gpu_mul(weight_count, new_weight_diff, weight, bottom_weight_diff + n * weight_count);
              break;
            case ConvolutionParameter_WeightOp_ADD:
              caffe_gpu_axpy(weight_count, Dtype(1.0), new_weight_diff, weight_diff);
              caffe_gpu_memcpy(weight_count, new_weight_diff, bottom_weight_diff + n * weight_count);
              break;
            case ConvolutionParameter_WeightOp_COPY:
              caffe_gpu_memcpy(weight_count, new_weight_diff, bottom_weight_diff + n * weight_count);
              break;
            default:
              LOG(FATAL) << "Unknown weight operation.";
          }
          // //mul
          // caffe_gpu_mul(weight_count, new_weight_diff, bottom_weight + n * weight_count, new_weight);
          // caffe_gpu_axpy(weight_count, Dtype(1.0), new_weight, weight_diff);
          // caffe_gpu_mul(weight_count, new_weight_diff, weight, bottom_weight_diff + n * weight_count);
          // // //add
          // caffe_gpu_axpy(weight_count, Dtype(1.0), new_weight_diff, weight_diff);
          // caffe_gpu_memcpy(weight_count, new_weight_diff, bottom_weight_diff + n * weight_count);
          // //cpy
          // caffe_gpu_memcpy(weight_count, new_weight_diff, bottom_weight_diff + n * weight_count);
        }
      }
    }
  }
  else{
    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();
      // Bias gradient, if necessary.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
        }
      }
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }

  }



  
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
