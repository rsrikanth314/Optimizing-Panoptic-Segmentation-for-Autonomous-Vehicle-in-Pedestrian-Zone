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
/*!\file    projects/TensorflowWrapper/Source/mDirectoryImagePublisher.h
 *
 * \author  Jakub Pawlak
 *
 * \date    2019-03-05
 *
 * \brief Contains mDirectoryImagePublisher
 *
 * \b mDirectoryImagePublisher
 *
 * Publishes images as tImage from a given directory for further processing.
 *
 */
//----------------------------------------------------------------------
#ifndef __libraries__machine_learning_appliance__utils__mDirectoryImagePublisher_h__
#define __libraries__machine_learning_appliance__utils__mDirectoryImagePublisher_h__

#include "plugins/structure/tModule.h"



//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>

#include "rrlib/coviroa/opencv_image_io.h"
#include "rrlib/coviroa/image_definitions.h"
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
namespace utils
{


//----------------------------------------------------------------------
// Forward declarations / typedefs / enums
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Class declaration
//----------------------------------------------------------------------
//! SHORT_DESCRIPTION
/*!
 * Publishes images as tImage from a given directory for further processing.
 */
class mDirectoryImagePublisher : public structure::tModule
{

  //----------------------------------------------------------------------
  // Ports (These are the only variables that may be declared public)
  //----------------------------------------------------------------------
public:

  tParameter<bool> activate;
  tOutput<rrlib::coviroa::tImage> out_image;
  std::string directory;


  //----------------------------------------------------------------------
  // Public methods and typedefs
  //----------------------------------------------------------------------
public:

  mDirectoryImagePublisher(core::tFrameworkElement *parent, const std::string &name = "DirectoryImagePublisher");

  //----------------------------------------------------------------------
  // Protected methods
  //----------------------------------------------------------------------
protected:

  /*! Destructor
   *
   * The destructor of modules is declared protected to avoid accidental deletion. Deleting
   * modules is already handled by the framework.
   */
  ~mDirectoryImagePublisher();

  //----------------------------------------------------------------------
  // Private fields and methods
  //----------------------------------------------------------------------
private:

  std::vector<std::string> file_list;
  std::vector<std::string>::iterator itr;

  virtual void Update() override;

};

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}




#endif
