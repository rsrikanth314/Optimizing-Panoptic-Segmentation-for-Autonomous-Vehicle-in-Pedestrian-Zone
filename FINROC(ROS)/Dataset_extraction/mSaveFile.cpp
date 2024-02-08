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
/*!\file    projects/data_AR/mSaveFile.cpp
 *
 * \author  Srikanth Reddy Yalamakuru
 *
 * \date    2021-04-09
 *
 */
//----------------------------------------------------------------------
#include "projects/Dataset_extraction/mSaveFile.h"
#include <ostream>
#include<string>
using namespace std;
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
runtime_construction::tStandardCreateModuleAction<mSaveFile> cCREATE_ACTION_FOR_M_SAVEFILE("SaveFile");

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// mSaveFile constructor
//----------------------------------------------------------------------
mSaveFile::mSaveFile(core::tFrameworkElement *parent, const std::string &name) :
  tModule(parent, name, true),
  par_start_save("Start Save", false, "start_save"),
  par_fram_length("Fram Length", 45, "fram_length"),
  start_save_(true),
  count_img_(0),
  // upperclass_(0),
  // lowerclass_(0),
  // ped_id_(0),
  count_frames_(0),
  check_(false)

{

}
//----------------------------------------------------------------------
// mSaveFile destructor
//----------------------------------------------------------------------
mSaveFile::~mSaveFile()

{}

//----------------------------------------------------------------------
// mSaveFile OnStaticParameterChange
//----------------------------------------------------------------------
void mSaveFile::OnStaticParameterChange()
{
  //if (this->static_parameter_1.HasChanged())
  {
    //As this static parameter has changed, do something with its value!
  }
}

//----------------------------------------------------------------------
// mSaveFile OnParameterChange
//----------------------------------------------------------------------
void mSaveFile::OnParameterChange()
{
	if (this->par_start_save.HasChanged())
			this->start_save_=this->par_start_save.Get();
	// if (this->par_upper_activity_class.HasChanged())
	// 		this->upperclass_=this->par_upper_activity_class.Get();
	// if (this->par_lower_activity_class.HasChanged())
	// 		this->lowerclass_=this->par_lower_activity_class.Get();
	// if (this->par_ped_id.HasChanged())
	// 		this->ped_id_=this->par_ped_id.Get();

	// if(this->par_U_calling.HasChanged())
	// {
	// 	if(this->par_U_calling.Get()==true)
	// 				this->par_upper_activity_class.Set(1);
	// }
	// if(this->par_U_texting.HasChanged())
	// {
	// 	if(this->par_U_texting.Get())
	// 			this->par_upper_activity_class.Set(2);
	// }
	// if(this->par_U_none.HasChanged())
	// {
	// 	if(this->par_U_none.Get())
	// 		this->par_upper_activity_class.Set(3);
	// }
	// if(this->par_U_waving.HasChanged())
	// {
	// 	if(this->par_U_waving.Get())
	// 		this->par_upper_activity_class.Set(4);
	// }
	// if(this->par_L_parallel_towards.HasChanged())
	// {
	// 	if(this->par_L_parallel_towards.Get())
	// 		this->par_lower_activity_class.Set(5);
	// }
	// if(this->par_L_parallel_away.HasChanged())
	// {
	// 	if(this->par_L_parallel_away.Get())
	// 			this->par_lower_activity_class.Set(6);
	// }
	// if(this->par_L_left_perpendicular.HasChanged())
	// {
	// 	if(this->par_L_left_perpendicular.Get())
	// 			this->par_lower_activity_class.Set(7);
	// }
	// if(this->par_L_right_perpendicular.HasChanged())
	// {
	// 	if(this->par_L_right_perpendicular.Get())
	// 				this->par_lower_activity_class.Set(8);
	// }
	// if(this->par_L_standing.HasChanged())
	// {
	// 	if(this->par_L_standing.Get())
	// 			this->par_lower_activity_class.Set(9);
	// }

}

//----------------------------------------------------------------------
// mSaveFile Update
//----------------------------------------------------------------------
void mSaveFile::Update()
{
//	if(this->waving_)
//	{
//		this->upperclass_=4;
//	}
//	if(this->calling_==true)
//	{
//		this->upperclass_=1;
//	}

	if (this->in_image.HasChanged())  // || in_pointcloud.HasChanged()||this->in_vecofvec_skeleton.HasChanged())
	{

		if(this->start_save_ && this->count_frames_<=this->par_fram_length.Get())
		{
			ofstream myFile;
			auto in_img_c = this->in_image.GetPointer();
			auto img_c = rrlib::coviroa::AccessImageAsMat(*in_img_c.Get());
			// auto in_cloud = this->in_pointcloud.GetPointer();
			// auto point_cloud = rrlib::distance_data::AccessDistanceDataAsMat(*in_cloud.Get());
			// auto skeleton = this->in_vecofvec_skeleton.GetPointer();
			//int upper = this->upperclass_;

			std::vector<int> compression_params;
			compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);

			bool result = false;
			try
			{
				// if(skeleton->size() > 0)
				// {
				///home/taf/finroc/sources/cpp/projects/data_AR/data/images/


				std::cout<<" Condition entered succesfulley"<<endl;

				result = cv::imwrite("/home/yalamaku/Documents/Detectron_2/Dataset/ZED_Cam_Extracted_data/images/" + std::to_string(this->count_img_) + ".png", img_c, compression_params);
				// myFile.open("/home/finroc/sources/cpp/projects/Data_extraction/data/dataset_zed.csv", ofstream::app);
				// if (myFile.is_open())
				// {
				// 	if (this->count<1)
				// 	{
				// 		myFile << "Image, skeleton data"<<";"<<endl;
				// 		myFile.close();
				// 	    this->count++;
				// 		//break;
				// 	}
				// 	else
				// 	{
				// 		/*To store the vector data of a skeleton in csv file with the image file name as well.
				// 		* The below line is for storing one skeletal data
				// 		* change to save for all skeletons*/
				// 			myFile << "/home/taf/finroc/sources/cpp/projects/data_AR/data/images/t" + std::to_string(this->count_img_) + ".png"<<'\t';
				// 			for (int i = 0; i < 18; i++)
				// 			{
				// 				for (int j=0; j<3; j++)
				// 					//myFile << i << ",";
				// 					myFile<<skeleton->at(0)[i][j]<<'\t';
				// 				if(i==17)
				// 					myFile<<upperclass_<<'\t'<<lowerclass_<<'\t'<<ped_id_<<endl;
				// 			}
				// 			//myFile << "/home/yogitha/Downloads/Test1/" + std::to_string(this->count_img_) + ".png"<<";"<<skeleton->at(0)[0][0]<<skeleton->at(0)[0]<<endl;


				// 		//myFile << "result, this->in_vel.Get(), in_steer.Get(),in_brake.Get()";
				// 		//myFile.open("test.csv");
				// 		//myFile<<skeleton;
				// 		myFile.close();
				// 		}
				// 	}
				// }

			}
			catch (const cv::Exception& ex)
			{
				fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
			}
			if (!result)
				printf("ERROR: Can't save PNG file.\n"); //printf("Saved PNG file with alpha data.\n");

			this->count_img_++;
			this->count_frames_++;
			this->check_=false;

		}
		else
		{
			this->count_frames_=0;
		    this->start_save_= false;
		}
		if (this->count_frames_ == 0 && !this->check_)
		{

			this->par_start_save.Set(this->start_save_);
			this->check_=true;
		}
	}

}
//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
