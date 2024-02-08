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
///*!\file    projects/onnx_ml_appliance/mOnnxInterface.cpp
// *
// * \author  Junejo FazeelAhmed
// *
// * \date    2021-09-09
// *
// */
////----------------------------------------------------------------------
#include "libraries/machine_learning_appliance/onnx/mOnnxInterface.h"
#include "rrlib/machine_learning_appliance/onnx/ort_utility/Constants.h"

//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------
#include <chrono>
#include <iostream>
#include <memory>
#include <cstring>
//----------------------------------------------------------------------
// Internal includes with ""
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Debugging
//----------------------------------------------------------------------
#include <cassert>

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

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// mOnnxInterface constructor
//----------------------------------------------------------------------
mOnnxInterface::mOnnxInterface(core::tFrameworkElement *parent, const std::string &name) : tModule(parent, name, false), // change to 'true' to make module's ports shared (so that ports in other processes can connect to its output and/or input ports)
  // par_path_to_graph("Path to Graph", this, std::string(getenv("FINROC_HOME")) + "/sources/cpp/libraries/machine_learning_appliance/onnx/examples/deeplabv3_surfacewater.onnx", "path_to_graph"),
  par_path_to_graph("Path to Graph", this, "/home/yalamaku/Documents/Thesis/Onnx_Models/Exported_onnx_models/sample_traced_model.onnx", "path_to_graph"),
  par_image_channels("Image Channels", this, 3, "image_channels"),
  par_image_height("Image Height", this, 256, "image_height"),
  par_image_width("Image Width", this, 256, "image_width"),
  par_number_of_classes("Number of Classes", this, rrlib::machine_learning_appliance::ort_utility::MSCOCO_NUM_CLASSES, "number_of_classes"),
  par_class_names("Name of the Classes", this, rrlib::machine_learning_appliance::ort_utility::MSCOCO_CLASSES, "class_names"),
  session_handler(par_number_of_classes.Get(), *par_path_to_graph.GetPointer() , 0,
                   std::vector<std::vector<int64_t>> {{par_image_channels.Get(), par_image_height.Get(), par_image_width.Get()}})

{
  session_handler.initClassNames(*par_class_names.GetPointer());
  FINROC_LOG_PRINT(DEBUG_VERBOSE_1, "Constructor of mOnnxInterface done");
  std::cout<<"Constructor of mOnnxInterface is done"<<std::endl;
}



//----------------------------------------------------------------------
// mOnnxInterface destructor
//----------------------------------------------------------------------
mOnnxInterface::~mOnnxInterface()
{}

//----------------------------------------------------------------------
// mOnnxInterface OnStaticParameterChange
//----------------------------------------------------------------------
//*void mOnnxInterface::OnStaticParameterChange()
// {}

//----------------------------------------------------------------------
// mOnnxInterface OnParameterChange
//----------------------------------------------------------------------
//void mOnnxInterface::OnParameterChange()
//{}

//----------------------------------------------------------------------
// mOnnxInterface Update
//----------------------------------------------------------------------

void mOnnxInterface::Update()
{

}

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}
//
