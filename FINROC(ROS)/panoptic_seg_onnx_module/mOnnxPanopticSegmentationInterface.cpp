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
/*!\file    libraries/machine_learning_appliance/onnx/mOnnxPanopticSegmentationInterface.cpp
 *
 * \author  Srikanth Reddy Yalamakuru
 *
 * \date    2023-11-23
 *
 */
//----------------------------------------------------------------------
#include "libraries/machine_learning_appliance/onnx/mOnnxPanopticSegmentationInterface.h"

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
#include <cstdint>
#include <vector>

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
#ifdef _LIB_FINROC_PLUGINS_RUNTIME_CONSTRUCTION_ACTIONS_PRESENT_
static const runtime_construction::tStandardCreateModuleAction<mOnnxPanopticSegmentationInterface> cCREATE_ACTION_FOR_M_ONNXPANOPTICSEGMENTATIONINTERFACE("OnnxPanopticSegmentationInterface");
#endif

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// mOnnxPanopticSegmentationInterface constructor
//----------------------------------------------------------------------
mOnnxPanopticSegmentationInterface::mOnnxPanopticSegmentationInterface(core::tFrameworkElement *parent, const std::string &name) :
  mOnnxInterface(parent, name) // change to 'true' to make module's ports shared (so that ports in other processes can connect to its output and/or input ports)
{
  std::cout<<" Setting Up mOnnxPanopticSegmentationInterface is done !"<<std::endl;
}

//----------------------------------------------------------------------
// mOnnxPanopticSegmentationInterface destructor
//----------------------------------------------------------------------
mOnnxPanopticSegmentationInterface::~mOnnxPanopticSegmentationInterface()
{}

//----------------------------------------------------------------------
// mOnnxPanopticSegmentationInterface OnStaticParameterChange
//----------------------------------------------------------------------
void mOnnxPanopticSegmentationInterface::OnStaticParameterChange()
{
}

//----------------------------------------------------------------------
// mOnnxPanopticSegmentationInterface OnParameterChange
//----------------------------------------------------------------------
void mOnnxPanopticSegmentationInterface::OnParameterChange()
{
}

//----------------------------------------------------------------------
// mOnnxPanopticSegmentationInterface Update
//----------------------------------------------------------------------
void mOnnxPanopticSegmentationInterface::Update()
{
  if (in_image.HasChanged())
  { 

    auto img_ptr = this->in_image.GetPointer();
    cv::Mat frame = rrlib::coviroa::AccessImageAsMat(*img_ptr); 

    // read image from file
    // std::string file = "/home/yalamaku/Documents/Thesis/Onnx_Models/Onnx_finroc_test_images/frame_1815.jpg";
    // cv::Mat frame = cv::imread(file);


    cv::imshow("Input Image BGR", frame);

    // Wait for a key press
    while (true) {
        int key = cv::waitKey(0);

        // Check if a key was pressed
        if (key >= 0) {
            break; // Exit the loop if a key was pressed
        }
    }

    // Close the OpenCV window
    cv::destroyAllWindows();

    // Convert BGR to RGB
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    cv::imshow("Input Image RGB", frame);

    // Wait for a key press
    while (true) {
        int key = cv::waitKey(0);

        // Check if a key was pressed
        if (key >= 0) {
            break; // Exit the loop if a key was pressed
        }
    }

    // Close the OpenCV window
    cv::destroyAllWindows();


    // std::cout<<"input image size: [" << frame.rows<< " " << frame.cols <<" " << frame.channels() << "]"<< std::endl;


    // Get the data type of image pixels
    int dataType = frame.type();
    std::cout<<"image data type:"<<dataType<<std::endl;


    // Access the first 5 elements of each channel
    auto range_end = std::min(5, frame.cols);

    // Iterate through the pixels in each channel
    for (int channel = 0; channel < frame.channels(); ++channel) {
        std::cout << "First 5 elements of Channel " << channel << ": ";
        for (int col = 0; col < range_end; ++col) {
            uint8_t value = frame.at<cv::Vec3b>(0, col)[channel];
            std::cout << static_cast<int>(value)<< " ";
        }
        std::cout << std::endl;
    }

    // std::cout<<"Input image mat:" <<" ["<< frame << "]" << std::endl;
  

    auto out_segmentation_buffer = out_segmentation_result.GetUnusedBuffer();
    out_segmentation_buffer->Resize(frame.cols, frame.rows, rrlib::coviroa::tImageFormat::eIMAGE_FORMAT_RGB24);
    cv::Mat out = rrlib::coviroa::AccessImageAsMat(*out_segmentation_buffer);

    if (frame.empty() || (frame.cols <= 1) || (frame.rows <= 1))
    {
      std::cout << "Failed to read input image" << std::endl;
    }
    else
    {
      std::cout<<"image read correctly with size; width: "<<frame.cols<<", height: "<<frame.rows<<", channels: "<<frame.channels()<<std::endl;

      std::vector<float> dst(this->par_image_channels.Get() * this->par_image_height.Get() * this->par_image_width.Get());

      auto masked_image = processOneFrame(frame, dst.data());

      masked_image.copyTo(out);
      out_segmentation_buffer.SetTimestamp(rrlib::time::Now());
      out_segmentation_result.Publish(out_segmentation_buffer);

      
    }
  }
}


//----------------------------------------------------------------------
// mOnnxSegmentationInterface preprocess
//----------------------------------------------------------------------
void mOnnxPanopticSegmentationInterface::preprocess(float* dst,                     //
    const float* src,                //
    const int64_t targetImgWidth,   //
    const int64_t targetImgHeight,  //
    const int numChannels) const
{

  std::cout<<"Starting preprocess with width: "<<targetImgWidth<<", height: "<<targetImgHeight<< ", channels: "<< numChannels;
  for (int c = 0; c < numChannels; ++c)
  {
    for (int i = 0; i < targetImgHeight; ++i)
    {
      for (int j = 0; j < targetImgWidth; ++j)
      {
        dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
          src[i * targetImgWidth * numChannels + j * numChannels + c];
      }
    }
  }
  std::cout<<" Pre-processing Done"<<std::endl;
}

//----------------------------------------------------------------------
// mOnnxSegmentationInterface preprocess
//----------------------------------------------------------------------
void mOnnxPanopticSegmentationInterface::preprocess(float* dst,                     //
    const cv::Mat& imgSrc,          //
    const int64_t targetImgWidth,   //
    const int64_t targetImgHeight,  //
    const int numChannels) const
{
  for (int i = 0; i < targetImgHeight; ++i)
  {
    for (int j = 0; j < targetImgWidth; ++j)
    {
      for (int c = 0; c < numChannels; ++c)
      {
        dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] = imgSrc.ptr<float>(i, j)[c];
      }
    }
  }
  std::cout<<" Pre-processing Done"<<std::endl;
  
}

cv::Mat mOnnxPanopticSegmentationInterface::postprocess(const std::vector<std::pair<int64_t*, std::vector<int64_t>>> inference_output) const
{
  
  const int64_t* output_data_ptr = inference_output[0].first;
  const std::vector<int64_t>& shape = inference_output[0].second;

  int width = shape[1];
  int height = shape[0];

  // Create a color segmentation map with 3 channels
    cv::Mat colorSegmentationMap(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // std::cout << "Color Segmentation Maps:\n" << colorSegmentationMap<< std::endl;

    // Iterate through the output array and set color values based on labels
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int64_t label = output_data_ptr[y * width + x];

            // Extract class and instance information from the label
            int originalClass = label / 1000;  // Divide by 1000 to get the original class id
            int instanceId = label % 1000;     // Get the instance id

            // Map originalClass to a color (default to white for unknown classes)
            cv::Vec3b color(0, 0, 0);

            // Add mappings for each class
            if (originalClass == 1) {
                color = cv::Vec3b(128, 128, 128);
            } else if (originalClass == 2) {
                color = cv::Vec3b(0, 165, 255);  
            } else if (originalClass == 3) {
                color = cv::Vec3b(165, 42, 42);  
            } else if (originalClass == 4) {
                color = cv::Vec3b(255, 192, 203); 
            } else if (originalClass == 5) {
                color = cv::Vec3b(0, 255, 0); 
            } else if (originalClass == 6) {
                color = cv::Vec3b(0, 128, 0); 
            }
            else if (originalClass == 7) {
                color = cv::Vec3b(128, 0, 128); 
            } else if (originalClass == 8) {
                color = cv::Vec3b(255, 0, 0);  
            } else if (originalClass == 9) {
                color = cv::Vec3b(0, 0, 255);  
            } else if (originalClass == 10) {
                color = cv::Vec3b(255, 0, 255);  
            }

            // Use instanceId to further differentiate instances within the same class
            color = color + cv::Vec3b(instanceId % 256, (instanceId * 37) % 256, (instanceId * 73) % 256);

            // Set the color value in the color segmentation map
            colorSegmentationMap.at<cv::Vec3b>(y, x) = color;

        }
    }

    // std::cout << "Color Segmentation Maps:\n" << colorSegmentationMap<< std::endl;

    cv::Size ColorSegMapSize =  colorSegmentationMap.size();

    std::cout<< "Color Segmentaion map size: ["<< ColorSegMapSize.height <<" "<< ColorSegMapSize.width<< "]"<< std::endl;

    return colorSegmentationMap;

}

cv::Mat mOnnxPanopticSegmentationInterface::processOneFrame(const cv::Mat& inputImg, float* dst)
{
  
  cv::Mat inputImg_;
  
  if (inputImg.channels() == 1)
  {
    std::vector<cv::Mat> vChannels;
    for (unsigned int c = 0; c < 3; c++)
    {
      vChannels.push_back(inputImg);
    }
    cv::merge(vChannels, inputImg_);
  }
  else
    inputImg.copyTo(inputImg_);

  cv::Mat processedImg;
  cv::resize(inputImg_, processedImg, cv::Size(this->par_image_width.Get(), this->par_image_height.Get()));

  // std::cout<<"resized image mat: "<< "["<< processedImg<< "]"<< std::endl;

  //Convert to Float 
  processedImg.convertTo(processedImg, CV_32FC3);

  // std::cout<<"resized image mat: "<< "["<< processedImg<< "]"<< std::endl;

  std::cout<<"inputImg size; width: "<<inputImg_.cols<<", height: "<<inputImg_.rows<<", channels: "<<inputImg_.channels()<<std::endl;
  std::cout<<"processedImg size; width: "<<processedImg.cols<<", height: "<<processedImg.rows<<", channels: "<<processedImg.channels()<<std::endl;

  preprocess(dst, processedImg, this->par_image_width.Get(), this->par_image_height.Get(), this->par_image_channels.Get());

  // To check the values of dst
  // const int size = this->par_image_channels.Get() * this->par_image_height.Get() * this->par_image_width.Get();

  // for (int i = 0; i < size; ++i) {
  //     std::cout << dst[i] << " ";
  // }
  // std::cout << std::endl;

  // To visualze dst using cv2
  // cv::Mat dummy_query = cv::Mat(this->par_image_height.Get(), this->par_image_width.Get(), CV_32F, dst);

  // // std::cout<<"dst dummy image mat: "<< "["<< dummy_query << "]"<< std::endl;

  // // Normalize the values to the range [0, 255]
  // cv::normalize(dummy_query, dummy_query, 0, 255, cv::NORM_MINMAX);

  // // Convert the matrix to 8-bit unsigned integer data type
  // cv::Mat dummy_query_uint8;
  // dummy_query.convertTo(dummy_query_uint8, CV_8U);

  // // Show the image using cv::imshow
  // cv::imshow("Dummy Query", dummy_query_uint8);

  // // Wait for a key press
  // while (true) {
  //     int key = cv::waitKey(0);

  //     // Check if a key was pressed
  //     if (key >= 0) {
  //         break; // Exit the loop if a key was pressed
  //     }
  // }

  // cv::destroyAllWindows();

  std::cout<<"dst image mat: "<< "["<< dst << "]"<< std::endl;

// /////////////////////////////////////////////////////////////

// Alternative pre-processing code that resizes and converts the image to the (C, H, W) fromate 
// Uncomment the above code to try this

//   cv::Mat ressized_image; 
//   cv::resize(inputImg, ressized_image, cv::Size(1820, 1024));

//   cv::imshow("resized image", ressized_image);

//   // Wait for a key press
//   while (true) {
//       int key = cv::waitKey(0);

//       // Check if a key was pressed
//       if (key >= 0) {
//           break; // Exit the loop if a key was pressed
//       }
//   }

//   // Close the OpenCV window
//   cv::destroyAllWindows();

//   // Convert the matrix to 32-bit floating point data type
//   ressized_image.convertTo(ressized_image, CV_32F);

//   // Transpose the image to (C, H, W) format
//   cv::Mat transposed = ressized_image.reshape(1, {3, 1024, 1820});
  
//   // Preprocess the image to (C, H, W) format
//   std::vector<float> input_data = transposed.reshape(1, {3 * 1024 * 1820});

//   cv::Mat dummy_query = cv::Mat(1024, 1820, CV_32F, input_data.data());

//   // std::cout<<"dst dummy image mat: "<< "["<< dummy_query << "]"<< std::endl;

//   // Normalize the values to the range [0, 255]
//   cv::normalize(dummy_query, dummy_query, 0, 255, cv::NORM_MINMAX);

//   // Convert the matrix to 8-bit unsigned integer data type
//   cv::Mat dummy_query_uint8;
//   dummy_query.convertTo(dummy_query_uint8, CV_8U);

//   // Show the image using cv::imshow
//   cv::imshow("Dummy Query", dummy_query_uint8);

//   // Wait for a key press
//   while (true) {
//       int key = cv::waitKey(0);

//       // Check if a key was pressed
//       if (key >= 0) {
//           break; // Exit the loop if a key was pressed
//       }
//   }

//   cv::destroyAllWindows();

//   /////////////////////////////////////////////////////////////////////////

  std::cout<<"Starting infernce"<< std::endl;

  std::vector<std::pair<int64_t*, std::vector<int64_t>>> inferenceOutput = this->session_handler({dst}); //{input_data.data()});

  std::cout<<"Infernecing Done !!"<<std::endl;
  std::cout<<"Infernce output vector length: "<<inferenceOutput.size()<<std::endl;


  for (const auto& output : inferenceOutput) {
    int64_t* data_ptr = output.first;
    const std::vector<int64_t>& shape = output.second;

    // Print the shape of the output tensor
    std::cout << "Output Tensor Shape: [";
    for (int64_t dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << "]" << std::endl;

    // Access the data inside the tensor using the data_ptr
    int64_t total_elements = 1;
    for (int64_t dim : shape) {
        total_elements *= dim;
    }

    std::cout<<"total_elements: "<<total_elements<<std::endl;

    std::cout << "Output Tensor Data: [";
    for (int64_t i = 0; i < total_elements; ++i) {
        std::cout << data_ptr[i] << " ";
    }
    std::cout << "]" << std::endl;
}

  int64_t* data_ptr = inferenceOutput[0].first;
  const std::vector<int64_t> shape = inferenceOutput[0].second;

  int width = shape[1];
  int height = shape[0];

  cv::Mat onnx_output_mat(height, width, CV_64F);

  // Access the data inside the tensor using the data_ptr and fill the matrix
  for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
          onnx_output_mat.at<double>(i, j) = data_ptr[i * width + j];
      }
  }

  // Normalize the values to the range [0, 255]
  cv::normalize(onnx_output_mat, onnx_output_mat, 0, 255, cv::NORM_MINMAX);

  cv::Mat onnx_output_mat_uint8;
  onnx_output_mat.convertTo(onnx_output_mat_uint8, CV_8U);

  // Visualize the output using cv::imshow
  cv::imshow("raw ONNX Output", onnx_output_mat_uint8);
  // cv::imwrite("/home/yalamaku/Documents/Thesis/Onnx_Models/Finroc_onnx_output_test/output_raw_2.jpg", onnx_output_mat_uint8);

  // Wait for a key press
  while (true) {
      int key = cv::waitKey(0);

      // Check if a key was pressed
      if (key >= 0) {
          break; // Exit the loop if a key was pressed
      }
  }

  // Close the OpenCV window
  cv::destroyAllWindows();

  // Visualize the result using the postprocess function
  cv::Mat resultImage = postprocess(inferenceOutput);

  // Display the original and processed images for verification

  cv::imshow("output image ", resultImage);
  // cv::imwrite("/home/yalamaku/Documents/Thesis/Onnx_Models/Finroc_onnx_output_test/output_pan_seg_2.jpg", resultImage);

  // Wait for a key press
  while (true) {
      int key = cv::waitKey(0);

      // Check if a key was pressed
      if (key >= 0) {
          break; // Exit the loop if a key was pressed
      }
  }

  // Close the OpenCV window
  cv::destroyAllWindows();

  return resultImage;
}
//----------------------------------------------------------------------
// End of namespace declaration
//----------------------------------------------------------------------
}
}
}
