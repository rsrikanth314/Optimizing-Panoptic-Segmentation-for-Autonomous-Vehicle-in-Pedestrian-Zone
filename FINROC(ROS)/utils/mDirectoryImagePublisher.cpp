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
/*!\file    projects/TensorflowWrapper/Source/mDirectoryImagePublisher.cpp
 *
 * \author  Jakub Pawlak
 *
 * \date    2019-03-05
 *
 */
//----------------------------------------------------------------------
#include "libraries/machine_learning_appliance/utils/mDirectoryImagePublisher.h"
#include "rrlib/coviroa/image_definitions.h"
#include "rrlib/coviroa/opencv_utils.h"
#include "rrlib/logging/messages.h"
#include "rrlib/coviroa/tImage.h"
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
namespace machine_learning_appliance
{
namespace utils
{

//----------------------------------------------------------------------
// Forward declarations / typedefs / enums
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Const values
//----------------------------------------------------------------------
runtime_construction::tStandardCreateModuleAction<mDirectoryImagePublisher> cCREATE_ACTION_FOR_M_DIRECTORYIMAGEPUBLISHER("DirectoryImagePublisher");

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// mDirectoryImagePublisher constructor
//----------------------------------------------------------------------
mDirectoryImagePublisher::mDirectoryImagePublisher(core::tFrameworkElement *parent, const std::string &name) :
  tModule(parent, name, false),
  activate(false),
  directory("/home/yalamaku/Documents/Thesis/Onnx_Models/Onnx_finroc_test_images")// change to 'true' to make module's ports shared (so that ports in other processes can connect to its output and/or input ports)

{
  dirent *dir_entry = NULL;
  unsigned int count = 0;

  RRLIB_LOG_PRINT(DEBUG, "Searching in ", directory);
  DIR* dir = opendir(directory.c_str());
  std::cout<<"Dir: "<< dir<< std::endl;

  if (dir)
  {
    while ((dir_entry = readdir(dir)))
    {
      std::cout<<"File_name: "<<dir_entry->d_name<<std::endl;
      // on some system the returned file type was 'unknown' DT_UNKNOWN
      // so it is only checked if the name does not start with a '.'
      //if (dir_entry->d_type == DT_REG) // entry is a regular file
      if ((dir_entry->d_type == DT_REG || dir_entry->d_type == DT_UNKNOWN) && dir_entry->d_name[0] != '.')
      {
        std::string filetype = std::string(dir_entry->d_name).substr(std::string(dir_entry->d_name).length() - 3, std::string(dir_entry->d_name).length());

        //Further checks for filetype:
        RRLIB_LOG_PRINT(DEBUG_VERBOSE_3, "filetype: ", filetype);
        if ((filetype == "jpg") || (filetype == "JPG") || (filetype == "png") || (filetype == "PNG"))
        {
          file_list.push_back(directory + "/" + std::string(dir_entry->d_name));
          count++;
          std::cout<<"count: "<< count<< std::endl;
        }
      }
    }
    RRLIB_LOG_PRINT(DEBUG_VERBOSE_2, "Found ", count, " file entries.");

    closedir(dir);
  }

  std::sort(file_list.begin(), file_list.end());

  itr = file_list.begin();

  std::cout<<"Constructor DirectoryImagePublisher is ready"<<std::endl;
}

//----------------------------------------------------------------------
// mDirectoryImagePublisher destructor
//----------------------------------------------------------------------
mDirectoryImagePublisher::~mDirectoryImagePublisher()
{}

//----------------------------------------------------------------------
// mDirectoryImagePublisher OnStaticParameterChange
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// mDirectoryImagePublisher OnParameterChange
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// mDirectoryImagePublisher Update
//----------------------------------------------------------------------
void mDirectoryImagePublisher::Update()
{
  if (activate.Get() && itr != file_list.end())
  {

    rrlib::coviroa::tImage imageToPublish(672, 376, rrlib::coviroa::eIMAGE_FORMAT_BGR24, 0); //Default consturctor is not exposed, need to change LoadImage function to properly resize
    std::cout << (*itr).c_str() << std::endl;
    if (itr->find(".txt") == std::string::npos)
    {
      rrlib::coviroa::LoadImage(itr->c_str(), imageToPublish);

      cv::Mat frame = rrlib::coviroa::AccessImageAsMat(imageToPublish); 
      

      std::string fn = "/home/yalamaku/Documents/Thesis/Onnx_Models/Onnx_finroc_test_images/34.jpg";
      cv::Mat I = cv::imread(fn);
      cv::imshow("testing1", I);
      cv::waitKey(0);
      

      data_ports::tPortDataPointer<rrlib::coviroa::tImage> out_image_buff = this->out_image.GetUnusedBuffer();

      
      out_image_buff->Resize(imageToPublish.GetWidth(), imageToPublish.GetHeight(),rrlib::coviroa::eIMAGE_FORMAT_BGR32);
      //    *(out_image_buff->GetImagePtr()) = *imageToPublish.GetImagePtr();
      cv::Mat out = rrlib::coviroa::AccessImageAsMat(*out_image_buff);
      //out_image_buff->ConvertFrom(imageToPublish);
      
      I.copyTo(out);
      cv::imshow("testing", out);
      cv::waitKey(0);
      out_image.Publish(out_image_buff);
    }
    else
    {
      RRLIB_LOG_PRINT(DEBUG, "Failed to laod image: ", *itr);
    }
    itr++;
    RRLIB_LOG_PRINT(DEBUG, *itr);

  }

}

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}
