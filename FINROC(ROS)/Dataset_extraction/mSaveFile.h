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
/*!\file    projects/data_AR/mSaveFile.h
 *
 * \author  Srikanth Reddy Yalamakuru
 *
 * \date    2021-04-09
 *
 * \brief Contains mSaveFile
 *
 * \b mSaveFile
 *
 * To save the image files of the skeleton from the ZED camera.
 *
 */
//----------------------------------------------------------------------
#ifndef __projects__Dataset_extraction__mSaveFile_h__
#define __projects__Dataset_extraction__mSaveFile_h__


#include "plugins/structure/tModule.h"

//----------------------------------------------------------------------
// External includes (system with <>, local with "")
//----------------------------------------------------------------------

#include <gflags/gflags.h>
#include <unistd.h>
//-------------------
//#include "libraries/unreal/converters/mUVehicleInput.h"

//----------------------------------------------------------------------
// Internal includes with ""
//----------------------------------------------------------------------
//#include <opencv2/opencv.hpp>
#include "rrlib/coviroa/tImage.h"
#include "rrlib/coviroa/opencv_utils.h"
#include <fstream>
#include "rrlib/distance_data/utils_opencv.h"
#include "rrlib/distance_data/tDistanceData.h"
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
// Class declaration
//----------------------------------------------------------------------
//! SHORT_DESCRIPTION
/*!
 * To save the image files of the skeleton from the ZED camera.
 */
class mSaveFile : public structure::tModule
{

//----------------------------------------------------------------------
// Ports (These are the only variables that may be declared public)
//----------------------------------------------------------------------
public:

	tInput<rrlib::coviroa::tImage> in_image;
	
  // tInput<rrlib::distance_data::tDistanceData> in_pointcloud;
	// tInput<std::vector<std::vector<rrlib::math::tVec3f>>> in_vecofvec_skeleton;

	//tInput<std::vector<rrlib::math::tVec3f>> my_vertices_vec;
	//tOutput<std::vector<std::vector<rrlib::math::tVec3f>>> out_vecofvec_skeleton;
	
  tParameter<bool> par_start_save;
	// tParameter<int> par_upper_activity_class;
	// tParameter<int> par_lower_activity_class;
	// tParameter<int> par_ped_id;
	tParameter<int> par_fram_length;

	//Activities
	// tParameter<bool> par_U_calling;
	// tParameter<bool> par_U_texting;
	// tParameter<bool> par_U_none;
	// tParameter<bool> par_U_waving;
	// tParameter<bool> par_L_parallel_towards;
	// tParameter<bool> par_L_parallel_away;
	// tParameter<bool> par_L_left_perpendicular;
	// tParameter<bool> par_L_right_perpendicular;
	// tParameter<bool> par_L_standing;


//----------------------------------------------------------------------
// Public methods and typedefs
//----------------------------------------------------------------------
public:

  mSaveFile(core::tFrameworkElement *parent, const std::string &name = "SaveFile");

//----------------------------------------------------------------------
// Protected methods
//----------------------------------------------------------------------
protected:

  /*! Destructor
   *
   * The destructor of modules is declared protected to avoid accidental deletion. Deleting
   * modules is already handled by the framework.
   */
  virtual ~mSaveFile();

//----------------------------------------------------------------------
// Private fields and methods
//----------------------------------------------------------------------
private:

  bool start_save_;
  int count_img_;
  int count=0;
  int counter = 0;
  // int upperclass_ ;
  // int lowerclass_;
  // int ped_id_;
  int count_frames_;
  bool check_;

//  bool calling_;
//  bool waving_;
//  bool texting_;
//  bool noact_;
//  bool parallel_towards_;
//  bool parallel_away;
//  bool perpendicular_left_;
//  bool perpendicular_right_;
//  bool standing_;
  //Here is the right place for your variables. Replace this line by your declarations!

  virtual void OnStaticParameterChange() override;   //Might be needed to process static parameters. Delete otherwise!

  virtual void OnParameterChange() override;   //Might be needed to react to changes in parameters independent from Update() calls. Delete otherwise!

  virtual void Update() override;

};

//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}



#endif
