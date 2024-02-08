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
/*!\file    projects/Dataset_extraction/gSaveFile.cpp
 *
 * \author  Srikanth Reddy Yalamakuru
 *
 * \date    2023-06-08
 *
 */
//----------------------------------------------------------------------
#include "projects/Dataset_extraction/gSaveFile.h"
#include "projects/Dataset_extraction/mSaveFile.h"

//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------

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
namespace Dataset_extraction
{

//----------------------------------------------------------------------
// Forward declarations / typedefs / enums
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Const values
//----------------------------------------------------------------------
#ifdef _LIB_FINROC_PLUGINS_RUNTIME_CONSTRUCTION_ACTIONS_PRESENT_
static const runtime_construction::tStandardCreateModuleAction<gSaveFile> cCREATE_ACTION_FOR_G_SAVEFILE("SaveFile");
#endif

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// gSaveFile constructor
//----------------------------------------------------------------------
gSaveFile::gSaveFile(core::tFrameworkElement *parent, const std::string &name,
                     const std::string &structure_config_file) :
  tGroup(parent, name, structure_config_file, true) // change to 'true' to make group's ports shared (so that ports in other processes can connect to its output and/or input ports)
{
  // initialize modules
  mSaveFile* save_file = new mSaveFile(this);
  
 // in_image.ConnectTo(¨https//:Dataset_extraction/SaveFile/Input/Image¨)

  in_image.ConnectTo(save_file->in_image);
}

//----------------------------------------------------------------------
// gSaveFile destructorda
//----------------------------------------------------------------------
gSaveFile::~gSaveFile()
{}

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
