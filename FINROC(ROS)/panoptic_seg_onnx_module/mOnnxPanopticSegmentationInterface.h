//
// You received this file as part of Finroc
// A framework for intelligent robot control
//
// Copyright (C) AG Robotersysteme TU Kaiserslautern
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
//----------------------------------------------------------------------
/*!\file    libraries/machine_learning_appliance/onnx/mOnnxPanopticSegmentationInterface.h
 *
 * \author  Srikanth Reddy Yalamakuru
 *
 * \date    2023-11-23
 *
 * \brief Contains mOnnxPanopticSegmentationInterface
 *
 * \b mOnnxPanopticSegmentationInterface
 *
 * Module whihc inherits from general Interface and inplements pre and post processing for panoptic segmeroks such as Panoptic_deeplab
 *
 */
//----------------------------------------------------------------------
#ifndef __libraries__machine_learning_appliance__onnx__mOnnxPanopticSegmentationInterface_h__
#define __libraries__machine_learning_appliance__onnx__mOnnxPanopticSegmentationInterface_h__

#include "plugins/structure/tModule.h"
#include "libraries/machine_learning_appliance/onnx/mOnnxInterface.h"
#include "rrlib/coviroa/tImage.h"
#include <cstdint>

//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------
// #include <opencv2/opencv.hpp>
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
 * Module whihc inherits from general Interface and inplements pre and post processing for panoptic segmeroks such as Panoptic_deeplab
 */
class mOnnxPanopticSegmentationInterface : public mOnnxInterface
{

//----------------------------------------------------------------------
// Ports (These are the only variables that may be declared public)
//----------------------------------------------------------------------
public:

  tOutput<rrlib::coviroa::tImage> out_segmentation_result;

//----------------------------------------------------------------------
// Public methods and typedefs
//----------------------------------------------------------------------
public:

  mOnnxPanopticSegmentationInterface(core::tFrameworkElement *parent, const std::string &name = "OnnxPanopticSegmentationInterface");


  void preprocess(float* dst,                     //
                  const float* src,               //
                  const int64_t targetImgWidth,   //
                  const int64_t targetImgHeight,  //
                  const int numChannels) const;

  void preprocess(float* dst,                     //
                  const cv::Mat& imgSrc,          //
                  const int64_t targetImgWidth,   //
                  const int64_t targetImgHeight,  //
                  const int numChannels) const;


  cv::Mat processOneFrame(const cv::Mat& inputImg, float* dst);

  cv::Mat postprocess(const std::vector<std::pair<int64_t*, std::vector<int64_t>>> inference_output) const;
//----------------------------------------------------------------------
// Protected methods
//----------------------------------------------------------------------
protected:

  /*! Destructor
   *
   * The destructor of modules is declared protected to avoid accidental deletion. Deleting
   * modules is already handled by the framework.
   */
  virtual ~mOnnxPanopticSegmentationInterface();

//----------------------------------------------------------------------
// Private fields and methods
//----------------------------------------------------------------------
private:

  virtual void OnStaticParameterChange() override;

  virtual void OnParameterChange() override; 

  virtual void Update() override;

};

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}



#endif
