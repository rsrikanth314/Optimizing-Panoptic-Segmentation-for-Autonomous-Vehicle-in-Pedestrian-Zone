//
////
//// You received this file as part of Finroc
//// A framework for intelligent robot control
////
//// Copyright (C) AG Robotersysteme TU Kaiserslautern
////
//// This program is free software; you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation; either version 2 of the License, or
//// (at your option) any later version.
////
//// This program is distributed in the hope that it will be useful,
//// but WITHOUT ANY WARRANTY; without even the implied warranty of
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//// GNU General Public License for more details.
////
//// You should have received a copy of the GNU General Public License along
//// with this program; if not, write to the Free Software Foundation, Inc.,
//// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
////
////----------------------------------------------------------------------
///*!\file    projects/onnx_ml_appliance/mOnnxInterface.h
// *
// * \author  Junejo FazeelAhmed
// *
// * \date    2021-09-09
// *
// * \brief Contains mOnnxInterface
// *
// * \b mOnnxInterface
// *
// * Wrapper for pretrained onnx models for use in finroc
// *
// */
////----------------------------------------------------------------------
#ifndef __libraries__machine_learning_appliance__onnx__mOnnxInterface_h__
#define __libraries__machine_learning_appliance__onnx__mOnnxInterface_h__

#include "plugins/structure/tModule.h"
//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------

#include <optional>
#include <string>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsuggest-attribute=format"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#else
#include <opencv2/opencv.hpp>
#endif
#include "rrlib/machine_learning_appliance/onnx/ort_utility/ort_utility.h"
#include "rrlib/coviroa/tImage.h"
//----------------------------------------------------------------------

// Internal includes with ""
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Namespace declaration
//----------------------------------------------------------------------
namespace finroc
{
namespace machine_learning_appliance
{
namespace onnx
{
//----------------------------------------------------------------------
// Forward declarations / typedefs / enums
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Class declaration
//----------------------------------------------------------------------
//! SHORT_DESCRIPTION
/*!
* Wrapper for pretrained onnx models for use in finroc
*/
class mOnnxInterface : public structure::tModule
{

//----------------------------------------------------------------------
// Ports (These are the only variables that may be declared public)
//----------------------------------------------------------------------
public:

  tInput<rrlib::coviroa::tImage> in_image;

  tParameter<std::string> par_path_to_graph;
  tParameter<int64_t> par_image_channels;
  tParameter<int64_t> par_image_height;
  tParameter<int64_t> par_image_width;
  tParameter<int64_t> par_number_of_classes;
  tParameter<std::vector<std::string>> par_class_names;

//----------------------------------------------------------------------
// Public methods and typedefs
//----------------------------------------------------------------------
public:

  mOnnxInterface(core::tFrameworkElement *parent, const std::string &name = "OnnxInterface");

  /*
  virtual void preprocess(float* dst,
                          const unsigned char* src,
                          const int64_t targetImgWidth,
                          const int64_t targetImgHeight,
                          const int numChannels) const = 0;

  virtual void preprocess(float* dst,                     //
                  const cv::Mat& imgSrc,          //
                  const int64_t targetImgWidth,   //
                  const int64_t targetImgHeight,  //
                  const int numChannels) const;

  virtual void postprocess(const std::vector<std::pair<float*, std::vector<int64_t>>> inference_output,
                           std::vector<rrlib::machine_learning_appliance::tMLStringDetection2D<float>>* detections) const = 0;

  virtual cv::Mat postprocess(const std::vector<std::pair<float*, std::vector<int64_t>>> inference_output) const;

  virtual void processOneFrame(std::vector<rrlib::machine_learning_appliance::tMLStringDetection2D<float>>* detection_results,
                               const cv::Mat& inputImg, float* dst) const = 0;

  virtual cv::Mat processOneFrame(const cv::Mat& inputImg) const;
  */
//----------------------------------------------------------------------
// Protected methods
//----------------------------------------------------------------------
protected:

  /*! Destructor
   *
   * The destructor of modules is declared protected to avoid accidental deletion. Deleting
   * modules is already handled by the framework.
   */
  virtual ~mOnnxInterface();

  rrlib::machine_learning_appliance::ort_utility::ImageRecognitionOrtSessionHandlerBase session_handler;

//----------------------------------------------------------------------
// Private fields and methods
//----------------------------------------------------------------------
private:

  virtual void Update() override;

};

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}
#endif
//
