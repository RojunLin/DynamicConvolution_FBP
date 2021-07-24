#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void ImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // if (prefetch_current_) {
  //   prefetch_free_.push(prefetch_current_);
  // }
  // prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // // Reshape to loaded data.
  // top[0]->ReshapeLike(prefetch_current_->data_);
  // top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  // if (this->output_labels_) {
  //   // Reshape to loaded labels.
  //   top[1]->ReshapeLike(prefetch_current_->label_);
  //   top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  // }
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  Blob<Dtype>* top_data = top[0];
  Dtype* top_label = top[1]->mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = top_data->offset(item_id);
    this->transformed_data_.set_cpu_data(top_data->mutable_cpu_data() + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    top_label[item_id] = lines_[lines_id_].second;

    // add extra information such as ethnic, gender
    if (!this->layer_param_.image_data_param().reference()){
      Dtype* top_extra = top[2]->mutable_cpu_data();
      std::string filename(lines_[lines_id_].first);
      if (filename.find('f') != std::string::npos) {
        top_extra[item_id * 2] = 1;     
      } else if (filename.find('m') != std::string::npos) {
        top_extra[item_id * 2] = -1;
      } else {
        LOG(ERROR) << "Filename error!";
      }
      if (filename.find('y') != std::string::npos) { 
        top_extra[item_id * 2 + 1] = 1;
      } else if (filename.find('w') != std::string::npos) {
        top_extra[item_id * 2 + 1] = -1;
      } else {
        LOG(ERROR) << "Filename error!";
      }
    }

    // // add extra information: ratios of facial landmarks
    // if (!this->layer_param_.image_data_param().reference()){
    //   Dtype* top_extra = top[2]->mutable_cpu_data();
    //   std::string ratio_folder = "/media/lin/Disk2/caffe-a/data/ratios/"; 
    //   std::string filename(lines_[lines_id_].first);
    //   filename.replace(filename.find("jpg"), 3, "txt");
    //   std::ifstream infile((ratio_folder + filename).c_str());
    //   string line;
    //   int index = 0;
    //   while (std::getline(infile, line)) {
    //     top_extra[item_id + index] = atof(line.c_str());
    //     // LOG(INFO) << top_extra[item_id + index] << " " << filename;
    //     index++;
    //   }
    //   if (index != top[2]->channels()) {
    //     LOG(ERROR) << "Filename error!";
    //   }
    // }

   // // add extra information such as ethnic, gender as extra input channels
   //  if (!this->layer_param_.image_data_param().reference()){
   //    Dtype* top_extra = top[2]->mutable_cpu_data();
   //    std::string filename(lines_[lines_id_].first);
   //    if (filename.find('f') != std::string::npos) {
   //      for (int h = 0; h < 224; h++) {
   //        for(int w = 0; w < 224; w++) {
   //          top_extra[(item_id * 2 * 224 + h) * 224 + w] = 1;
   //        }
   //      } 
   //    } else if (filename.find('m') != std::string::npos) {
   //      for (int h = 0; h < 224; h++) {
   //        for(int w = 0; w < 224; w++) {
   //          top_extra[(item_id * 2 * 224 + h) * 224 + w] = -1;
   //        }
   //      } 
   //    } else {
   //      LOG(ERROR) << "Filename error!";
   //    }
   //    if (filename.find('y') != std::string::npos) { 
   //      for (int h = 0; h < 224; h++) {
   //        for(int w = 0; w < 224; w++) {
   //          top_extra[((item_id * 2 + 1) * 224 + h) * 224 + w] = 1;
   //        }
   //      } 
   //    } else if (filename.find('w') != std::string::npos) {
   //      for (int h = 0; h < 224; h++) {
   //        for(int w = 0; w < 224; w++) {
   //          top_extra[((item_id * 2 + 1) * 224 + h) * 224 + w] = -1;
   //        }
   //      }
   //    } else {
   //      LOG(ERROR) << "Filename error!";
   //    }
   //  }


    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}
INSTANTIATE_LAYER_GPU_FORWARD(ImageDataLayer);
}  // namespace caffe
#endif  // USE_OPENCV
