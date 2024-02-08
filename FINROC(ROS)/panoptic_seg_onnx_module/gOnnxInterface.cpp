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
/*!\file    projects/onnx_ml_appliance/gOnnxInterface.cpp
 *
 * \author  Junejo FazeelAhmed
 *
 * \date    2021-08-30
 *
 */
//----------------------------------------------------------------------
#include "libraries/machine_learning_appliance/onnx/gOnnxInterface.h"

//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------
#include "libraries/machine_learning_appliance/onnx/mOnnxPanopticSegmentationInterface.h"
#include "rrlib/machine_learning_appliance/onnx/ort_utility/Constants.h"
//----------------------------------------------------------------------
// Internal includes with ""
//----------------------------------------------------------------------
#include "libraries/machine_learning_appliance/onnx/mOnnxSegmentationInterface.h"
#include "libraries/machine_learning_appliance/utils/mDirectoryImagePublisher.h"
//----------------------------------------------------------------------
// Debugging
//----------------------------------------------------------------------
#include <cassert>
#include <iostream>

//----------------------------------------------------------------------
// Namespace usage
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
// Const values
//----------------------------------------------------------------------
static runtime_construction::tStandardCreateModuleAction<gOnnxInterface> cCREATE_ACTION_FOR_G_ONNXINTERFACE("OnnxInterface");

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// gOnnxInterface constructor
//----------------------------------------------------------------------
gOnnxInterface::gOnnxInterface(core::tFrameworkElement *parent, const std::string &name,
                               const std::string &structure_config_file) :
  tGroup(parent, name, structure_config_file, false) // change to 'true' to make group's ports shared (so that ports in other processes can connect to its output and/or input ports)
{

  // auto object_segmentation_module = new finroc::machine_learning_appliance::onnx::mOnnxSegmentationInterface(this);
  // object_segmentation_module->par_path_to_graph.Set(std::string(getenv("FINROC_HOME")) + "/sources/cpp/libraries/machine_learning_appliance/onnx/examples/deeplabv3_surfacewater.onnx");
  // object_segmentation_module->par_image_channels.Set(3);
  // object_segmentation_module->par_image_height.Set(256);
  // object_segmentation_module->par_image_width.Set(256);
  // object_segmentation_module->par_number_of_classes.Set(rrlib::machine_learning_appliance::ort_utility::SURFACEWATER_NUM_CLASSES);
  // object_segmentation_module->par_class_names.Set(rrlib::machine_learning_appliance::ort_utility::SURFACEWATER_CLASSES);

  auto panoptic_segmentation_module = new finroc::machine_learning_appliance::onnx::mOnnxPanopticSegmentationInterface(this);
  panoptic_segmentation_module->par_image_channels.Set(3);
  panoptic_segmentation_module->par_image_height.Set(1024);
  panoptic_segmentation_module->par_image_width.Set(1820);
  panoptic_segmentation_module->par_number_of_classes.Set(rrlib::machine_learning_appliance::ort_utility::PANOPTICSEG_TUKUNI_NUM_CLASSES);
  panoptic_segmentation_module->par_class_names.Set(rrlib::machine_learning_appliance::ort_utility::PANOPTICSEG_TUKUNI_CLASSES);


  // auto image_publisher = new finroc::machine_learning_appliance::utils::mDirectoryImagePublisher(this);
  // image_publisher->directory = "/home/a_vierling/Temp_Dataset/onnx_test";
  // image_publisher->directory = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/test/test_images";
  // std::cout<<"image_dir:"<<image_publisher->directory<<std::endl;

  // image_publisher->out_image.ConnectTo(panoptic_segmentation_module->in_image);

}

//----------------------------------------------------------------------
// gOnnxInterface destructor
//----------------------------------------------------------------------
gOnnxInterface::~gOnnxInterface()
{}

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}
