/*
  CSCI 420 Computer Graphics, Computer Science, USC
  Assignment 2: Simulating a Roller Coaster.
  C/C++ starter code

  Student username: <psrathod>
*/

#include "openGLHeader.h"
#include "glutHeader.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "pipelineProgram.h"
#include "vbo.h"
#include "vao.h"
#include <glm/glm.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>

#include <iostream>
#include <cstring>
#include <vector>

#if defined(WIN32) || defined(_WIN32)
  #ifdef _DEBUG
    #pragma comment(lib, "glew32d.lib")
  #else
    #pragma comment(lib, "glew32.lib")
  #endif
#endif

#if defined(WIN32) || defined(_WIN32)
  char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
  char shaderBasePath[1024] = "../openGLHelper";
#endif

using namespace std;


// Width and height of the OpenGL window, in pixels.
int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 Homework 2";
int frameCount = 0, countSS = 0;
// Stores the image loaded from disk.
ImageIO * heightmapImage;

// Number of vertices in the single triangle (starter code).
int numVertices;
int dispMode = 1; // default display points


int cam = 0;
vector <float> positions, colors, tangents, upVectors, binormals;
vector <glm::vec3 * > squares, otherSquares;
// vector <glm::vec3> binormals;
// CSCI 420 helper classes.
OpenGLMatrix matrix;
PipelineProgram * pipelineProgram = nullptr;
PipelineProgram * texturePipeline = nullptr;


VBO * vboVertices = nullptr;
VBO * vboColors = nullptr;
VAO * vao = nullptr;

VBO * vboVerticesPath = nullptr;
VBO * vboColorsPath = nullptr;
VAO * vaoPath = nullptr;

VBO * vboVerticesTex = nullptr;
VBO * vboColorsTex = nullptr;
VAO * vaoTex = nullptr;

GLuint texHandle;

// Represents one spline control point.
struct Point 
{
  double x, y, z;
};

// Contains the control points of the spline.
struct Spline 
{
  int numControlPoints;
  Point * points;
} spline;

void loadSpline(char * argv) 
{
  FILE * fileSpline = fopen(argv, "r");
  if (fileSpline == NULL) 
  {
    printf ("Cannot open file %s.\n", argv);
    exit(1);
  }

  // Read the number of spline control points.
  fscanf(fileSpline, "%d\n", &spline.numControlPoints);
  printf("Detected %d control points.\n", spline.numControlPoints);

  // Allocate memory.
  spline.points = (Point *) malloc(spline.numControlPoints * sizeof(Point));
  // Load the control points.
  for(int i=0; i<spline.numControlPoints; i++)
  {
    if (fscanf(fileSpline, "%lf %lf %lf", 
           &spline.points[i].x, 
	   &spline.points[i].y, 
	   &spline.points[i].z) != 3)
    {
      printf("Error: incorrect number of control points in file %s.\n", argv);
      exit(1);
    }
  }
}

// Multiply C = A * B, where A is a m x p matrix, and B is a p x n matrix.
// All matrices A, B, C must be pre-allocated (say, using malloc or similar).
// The memory storage for C must *not* overlap in memory with either A or B. 
// That is, you **cannot** do C = A * C, or C = C * B. However, A and B can overlap, and so C = A * A is fine, as long as the memory buffer for A is not overlaping in memory with that of C.
// Very important: All matrices are stored in **column-major** format.
// Example. Suppose 
//      [ 1 8 2 ]
//  A = [ 3 5 7 ]
//      [ 0 2 4 ]
//  Then, the storage in memory is
//   1, 3, 0, 8, 5, 2, 2, 7, 4. 
void MultiplyMatrices(int m, int p, int n, const double * A, const double * B, double * C)
{
  for(int i=0; i<m; i++)
  {
    for(int j=0; j<n; j++)
    {
      double entry = 0.0;
      for(int k=0; k<p; k++)
        entry += A[k * m + i] * B[j * p + k];
      C[m * j + i] = entry;
    }
  }
}

int initTexture(const char * imageFilename, GLuint textureHandle)
{
  // Read the texture image.
  ImageIO img;
  ImageIO::fileFormatType imgFormat;
  ImageIO::errorType err = img.load(imageFilename, &imgFormat);

  if (err != ImageIO::OK) 
  {
    printf("Loading texture from %s failed.\n", imageFilename);
    return -1;
  }

  // Check that the number of bytes is a multiple of 4.
  if (img.getWidth() * img.getBytesPerPixel() % 4) 
  {
    printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
    return -1;
  }

  // Allocate space for an array of pixels.
  int width = img.getWidth();
  int height = img.getHeight();
  unsigned char * pixelsRGBA = new unsigned char[4 * width * height]; // we will use 4 bytes per pixel, i.e., RGBA

  // Fill the pixelsRGBA array with the image pixels.
  memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
  for (int h = 0; h < height; h++)
    for (int w = 0; w < width; w++) 
    {
      // assign some default byte values (for the case where img.getBytesPerPixel() < 4)
      pixelsRGBA[4 * (h * width + w) + 0] = 0; // red
      pixelsRGBA[4 * (h * width + w) + 1] = 0; // green
      pixelsRGBA[4 * (h * width + w) + 2] = 0; // blue
      pixelsRGBA[4 * (h * width + w) + 3] = 255; // alpha channel; fully opaque

      // set the RGBA channels, based on the loaded image
      int numChannels = img.getBytesPerPixel();
      for (int c = 0; c < numChannels; c++) // only set as many channels as are available in the loaded image; the rest get the default value
        pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
    }

  // Bind the texture.
  glBindTexture(GL_TEXTURE_2D, textureHandle);

  // Initialize the texture.
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

  // Generate the mipmaps for this texture.
  glGenerateMipmap(GL_TEXTURE_2D);

  // Set the texture parameters.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // Query support for anisotropic texture filtering.
  GLfloat fLargest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
  printf("Max available anisotropic samples: %f\n", fLargest);
  // Set anisotropic texture filtering.
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

  // Query for any errors.
  GLenum errCode = glGetError();
  if (errCode != 0) 
  {
    printf("Texture initialization error. Error code: %d.\n", errCode);
    return -1;
  }
  
  // De-allocate the pixel array -- it is no longer needed.
  delete [] pixelsRGBA;

  return 0;
}

// Write a screenshot to the specified filename.
void saveScreenshot(const char * filename)
{
  unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

  if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else cout << "Failed to save file " << filename << '.' << endl;

  delete [] screenshotData;
}

void idleFunc()
{
  // Do some stuff... 
  // For example, here, you can save the screenshots to disk (to make the animation).
  
  // Notify GLUT that it should call displayFunc.
  // commented out code to take 300 screenshots every (1 screenshot per 10 frames).
  // if (frameCount <= 300000){
  //   // cout << frameCount++ << endl;
  //   frameCount++;
  //     if (frameCount % 5 == 0 && countSS <=1000){
  //     char buf [10];
  //     sprintf(buf, "%03d.jpeg", countSS);

  //     std::string filename = "AnimationFrames/" + std::string(buf);

  //     char* concatenatedString = new char[filename.length() + 1]; // Allocate memory for the concatenated string
  //     strcpy(concatenatedString, filename.c_str()); // Copy the contents of the concatenated std::string to the char*
  //     saveScreenshot(concatenatedString);
  //     std::cout << concatenatedString << std::endl; // Output the concatenated string

  //     delete[] concatenatedString;
  //     // cout << frameCount << " - "  << buf << endl;
  //     // std::string fname = "" + countSS + ".jpeg";
      
  //     countSS++;
  //   }
  // }

  glutPostRedisplay();
}

void reshapeFunc(int w, int h)
{
  glViewport(0, 0, w, h);

  // When the window has been resized, we need to re-set our projection matrix.
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.LoadIdentity();
  // You need to be careful about setting the zNear and zFar. 
  // Anything closer than zNear, or further than zFar, will be culled.
  const float zNear = 0.1f;
  const float zFar = 10000.0f;
  const float humanFieldOfView = 60.0f;
  matrix.Perspective(humanFieldOfView, 1.0f * w / h, zNear, zFar);
}


void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27: // ESC key
      exit(0); // exit the program
    break;

    case 'x':
      // Take a screenshot.
      saveScreenshot("screenshot.jpg");
    break;
  }
}



void displayFunc()
{
  // This function performs the actual rendering.

  // First, clear the screen.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set up the camera position, focus point, and the up vector.
  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.LoadIdentity();
  // matrix.LookAt(0.0, 2.0, 5.0,
  //               0.0, 0.0, 0.0,
  //               0.0, 1.0, 0.0);
  
  matrix.LookAt(positions[cam] + upVectors[cam], positions[cam + 1] + upVectors[cam + 1], positions[cam + 2] + upVectors[cam + 2],
                (positions[cam] + tangents[cam]), (positions[cam + 1] + tangents[cam + 1]), (positions[cam + 2] + tangents[cam +2]),
                (upVectors[cam]), (upVectors[cam + 1]), (upVectors[cam + 2]));
  if (cam >= positions.size()){
    cam = 0;
    
  } else {
    // cout << cam << endl;
    // cout << positions[cam] << " " << positions[cam + 1] << " " << positions[cam + 2] << endl;
    cam += 6;
  }

  

  float view[16]; 
  matrix.GetMatrix(view); 

  float lightDirection[3] = {0, 1, 0}; // the “Sun” at noon 
  float viewLightDirection[4];
  float lightDirHomogeneous[4] = { lightDirection[0], lightDirection[1], lightDirection[2], 0.0 };
  // Multiply the light direction by the view matrix
  for (int i = 0; i < 3; ++i) {
      viewLightDirection[i] = 0.0f;
      for (int j = 0; j < 4; ++j) {
          viewLightDirection[i] += view[j * 4 + i] * lightDirHomogeneous[j];
      }
  }

  glm::vec3 viewLightDirectionGLM (viewLightDirection[0], viewLightDirection[1], viewLightDirection[2]);
  viewLightDirectionGLM = glm::normalize(viewLightDirectionGLM);
  viewLightDirection[0] = viewLightDirectionGLM.x;
  viewLightDirection[1] = viewLightDirectionGLM.y;
  viewLightDirection[2] = viewLightDirectionGLM.z;


  // Read the current modelview and projection matrices from our helper class.
  // The matrices are only read here; nothing is actually communicated to OpenGL yet.
  float n[16];
  float modelViewMatrix[16];

  matrix.SetMatrixMode(OpenGLMatrix::ModelView);
  matrix.GetNormalMatrix(n);
  matrix.GetMatrix(modelViewMatrix);
  
  float projectionMatrix[16];
  matrix.SetMatrixMode(OpenGLMatrix::Projection);
  matrix.GetMatrix(projectionMatrix);

  texturePipeline->Bind();
  texturePipeline->SetUniformVariableMatrix4fv("modelViewMatrix", GL_FALSE, modelViewMatrix);
  texturePipeline->SetUniformVariableMatrix4fv("projectionMatrix", GL_FALSE, projectionMatrix);
  // texturePipeline->Bind();
  glActiveTexture(GL_TEXTURE0);
  GLint textureLocation = glGetUniformLocation(texturePipeline->GetProgramHandle(), "textureImage");
  glUniform1i(textureLocation, 0); 
  
  
  glBindTexture(GL_TEXTURE_2D, texHandle);

  vaoTex->Bind();
 
  glDrawArrays(GL_TRIANGLES, 0, 6);

  float La[4] = { 0.2f, 0.2f, 0.2f, 1.0f }, Ld[4] = { 0.8f, 0.8f, 0.8f, 1.0f }, Ls[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
  float ka[4] = { 0.35f, 0.35f, 0.35f, 1.0f }, kd[4] = { 0.5f, 0.5f, 0.5f, 1.0f }, ks[4] = { 0.9f, 0.9f, 0.9f, 1.0f };
  float alpha = 2.0f;
  pipelineProgram->Bind();
  // Upload the modelview and projection matrices to the GPU. Note that these are "uniform" variables.
  // Important: these matrices must be uploaded to *all* pipeline programs used.
  // In hw1, there is only one pipeline program, but in hw2 there will be several of them.
  // In such a case, you must separately upload to *each* pipeline program.
  // Important: do not make a typo in the variable name below; otherwise, the program will malfunction.
  pipelineProgram->SetUniformVariableMatrix4fv("modelViewMatrix", GL_FALSE, modelViewMatrix);
  pipelineProgram->SetUniformVariableMatrix4fv("projectionMatrix", GL_FALSE, projectionMatrix);
  pipelineProgram->SetUniformVariableMatrix4fv("normalMatrix", GL_FALSE, n);

  pipelineProgram->SetUniformVariable4fv("ka", ka);
  pipelineProgram->SetUniformVariable4fv("kd", kd);
  pipelineProgram->SetUniformVariable4fv("ks", ks);
  pipelineProgram->SetUniformVariable4fv("La", La);
  pipelineProgram->SetUniformVariable4fv("Ld", Ld);
  pipelineProgram->SetUniformVariable4fv("Ls",  Ls);
  pipelineProgram->SetUniformVariable4fv("viewLightDirection", viewLightDirection);
  pipelineProgram->SetUniformVariablef("alpha", alpha);
  pipelineProgram->Bind();

  vaoPath->Bind();
  glDrawArrays(GL_TRIANGLES, 0, numVertices);
  glutSwapBuffers();
}

// returns the basis matrix used to compute the spline.
double * getBasisMatrix(double s){
  double * catmullRomMatrix = new double[16]{
    -s, (2*s), -s, 0,
    2-s, s-3, 0, 1,
    s-2, 3 - (2 * s), s, 0,
    s, -s, 0, 0 
  };
  // std::cout << catmullRomMatrix.data()[1] << endl;
  return catmullRomMatrix;
}

double * getUMatrix(int n){
  double u = n / 100.0;
  double* uMatrix = new double[4]{pow(u, 3), pow(u, 2), u, 1};
  return uMatrix;
}

// returns the control matrix used to compute the spline
double * getControlMatrix (int i){
  double* splineMatrix = new double[12] {
    spline.points[i].x, spline.points[i+1].x, spline.points[i + 2].x, spline.points[i + 3].x,
    spline.points[i].y, spline.points[i+1].y, spline.points[i + 2].y, spline.points[i + 3].y,
    spline.points[i].z, spline.points[i+1].z, spline.points[i + 2].z, spline.points[i + 3].z,
  };
  return splineMatrix;
}

// compute the rail crossSection vertex
glm::vec3 getSquareVertex (glm::vec3 pos, glm::vec3 nor, glm::vec3 bi){
  return pos + 0.1f * (nor + bi);
}

// get the cross section (4 points of the square) at a point on the spline
void getSquare (int n) {
  // cout << n << endl;
  glm::vec3 pos (positions[n], positions[n + 1], positions[n + 2]);
  glm::vec3 nor (upVectors[n], upVectors[n + 1], upVectors[n + 2]);
  glm::vec3 bi (binormals[n], binormals[n + 1], binormals[n + 2]);

  glm::vec3 V0 = getSquareVertex (pos, -nor, bi);
  glm::vec3 V1 = getSquareVertex (pos, nor, bi);
  glm::vec3 V2 = getSquareVertex (pos, nor, -bi);
  glm::vec3 V3 = getSquareVertex (pos, -nor, -bi);
  
  bi.x = bi.x/2;
  bi.y = bi.y/2;
  bi.z = bi.z/2;
  glm::vec3 *squareVertices = new glm::vec3[4]{V0+bi, V1+bi, V2+bi, V3+bi};
  squares.push_back(squareVertices);

  glm::vec3 *squareVertices2 = new glm::vec3[4]{V0-bi, V1-bi, V2-bi, V3-bi};
  otherSquares.push_back(squareVertices2);
}

// add the vertex of the rail cross section
void addVert(glm::vec3 Vert, vector<float> & rail){
  // cout << " TrigVert " << Vert.x << " " << Vert.y << " " << Vert.z << endl; 
  rail.push_back(Vert.x);
  rail.push_back(Vert.y);
  rail.push_back(Vert.z);
}


// for every triangle of the cross section - compute the normal that will be used as the color
void computeTriangleNormal(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, vector<float> & colors) {
    // Compute two edges of the triangle
    glm::vec3 edge1 = v2 - v1;
    glm::vec3 edge2 = v1 - v0;

    // Compute the cross product of the edges
    glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
    // cout<< " Color " << normal.x << " " << normal.y << " " << normal.z << endl;
    for (int i = 0; i<3 ; i ++){
      if (isnan(normal.x)){
        colors.push_back(0);
        colors.push_back(0);
        colors.push_back(0);
      } else {
        colors.push_back(normal.x);
        colors.push_back(normal.y);
        colors.push_back(normal.z);
      }
      
      colors.push_back(1.0f);
      // colors.push_back(0.0f);
    }
}

// add the cross section of the rail track
void crossSection (glm::vec3* sq1, glm::vec3* sq2, vector<float> & rail, vector<float> & colors){
  // right cross section
  addVert(sq2[0], rail); addVert(sq1[0], rail); addVert(sq2[1], rail);
  computeTriangleNormal (sq2[0], sq1[0], sq2[1], colors);
  addVert(sq1[1], rail); addVert(sq2[1], rail); addVert(sq1[0], rail);
  computeTriangleNormal (sq1[1], sq2[1], sq1[0], colors);

  // // top crossSection
  addVert(sq2[1], rail); addVert(sq1[1], rail); addVert(sq2[2], rail);
  computeTriangleNormal (sq2[1], sq1[1], sq2[2], colors);
  addVert(sq1[2], rail); addVert(sq2[2], rail); addVert(sq1[1], rail);
  computeTriangleNormal (sq1[2], sq2[2], sq1[1], colors);

  //  left cross section
  addVert(sq2[3], rail); addVert(sq1[2], rail); addVert(sq2[2], rail);
  computeTriangleNormal (sq2[3], sq1[2], sq2[2], colors);
  addVert(sq1[2], rail); addVert(sq2[3], rail); addVert(sq1[3], rail);
  computeTriangleNormal (sq1[2], sq2[3], sq1[3], colors);


  // bottom cross section
  addVert(sq2[0], rail); addVert(sq1[0], rail); addVert(sq2[3], rail);
  computeTriangleNormal (sq2[0], sq1[0], sq2[3], colors);
  addVert(sq1[3], rail); addVert(sq2[3], rail); addVert(sq1[0], rail);
  computeTriangleNormal (sq1[3], sq2[3], sq1[0], colors);
  
}

// function to assign the VBO for the rail track.
void makeCubeVBO(){
  // cout << " HI " << endl;
  vector <float> railTrig;
  vector <float> color;
  for (int i = 0; i < squares.size() - 1; i++){
    crossSection(squares[i], squares[i+1], railTrig, color);
    crossSection (otherSquares[i], otherSquares[i+1], railTrig, color);
  }
  numVertices = railTrig.size() / 3;
  vboVerticesPath = new VBO(railTrig.size()/3, 3, railTrig.data(), GL_STATIC_DRAW); // 3 values per position
  vboColorsPath = new VBO(railTrig.size()/3, 4, color.data(), GL_STATIC_DRAW); // 4 values per color
  vaoPath = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO.
  // Important: any typo in the shader variable name will lead to malfunction.
  vaoPath->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboVerticesPath, "position");

  // Set up the relationship between the "color" shader variable and the VAO.
  // Important: any typo in the shader variable name will lead to malfunction.
  vaoPath->ConnectPipelineProgramAndVBOAndShaderVariable(pipelineProgram, vboColorsPath, "normal");
 

}


// Iterate through all the tangents and compute the up vector used to properly orient the camera
void computeUpVector(){
  glm::vec3 arbitraryVector(0.0f, 0.0f, 1.0f);
  glm::vec3 firstTan(tangents[0], tangents[1], tangents[2]);
  
  glm::vec3 N0 = glm::normalize(glm::cross(arbitraryVector, firstTan));
  glm::vec3 Binormal = glm::normalize(glm::cross(firstTan, N0));

  upVectors.push_back(N0.x);
  upVectors.push_back(N0.y);
  upVectors.push_back(N0.z);

  binormals.push_back(Binormal.x);
  binormals.push_back(Binormal.y);
  binormals.push_back(Binormal.z);

  getSquare(0);
  for (int i = 3; i < tangents.size(); i+= 3){
    glm::vec3 tangent(tangents[i], tangents[i+1], tangents[i + 2]);

    glm::vec3 normal = glm::normalize(glm::cross(Binormal, tangent));
    Binormal = glm::normalize(glm::cross(tangent, normal));

    upVectors.push_back(normal.x);
    upVectors.push_back(normal.y);
    upVectors.push_back(normal.z);
    binormals.push_back(Binormal.x);
    binormals.push_back(Binormal.y);
    binormals.push_back(Binormal.z);

    getSquare(i);
  }

  cout << " NUM Cross Sections : " << squares.size() << endl;

}



// Using the current control matrix - compute the tangent
void computeTangent (int n, double * MC){
  double u = n / 100.0;
  // cout << "Tanget U = " << u << endl; 
  double* newUMatrix = new double[4]{3 * pow(u, 2), 2 * u, 1, 0};
  double * tangent = (double *) malloc(3 * sizeof(double));
  MultiplyMatrices(1, 4, 3, newUMatrix, MC, tangent);
  glm::vec3 tans(tangent[0], tangent[1], tangent[2]);
  // tans = glm::normalize(tans);
  tangents.push_back(tans.x);
  tangents.push_back(tans.y);
  tangents.push_back(tans.z);

}




void makeVBO(double * basis, double * control, vector<float> & positions, vector<float> & colors){
  double * mul1 = (double *) malloc(12 * sizeof(double));
  MultiplyMatrices(4, 4, 3, basis, control, mul1);
  for (int u = 0; u < 100; u++){
    // getUMatrix(u);
    double * mul2 = (double *) malloc(3 * sizeof(double));
    MultiplyMatrices(1, 4, 3, getUMatrix(u), mul1, mul2);
    computeTangent(u, mul1);
    // glm::vec3 *square1 = getSquare (mul2, tangents.size() - 1);
    // cout << "X Y Z " << endl;
    // cout << mul2[0]<< " " << mul2[1]<< " " << mul2[2]<< " " << endl;
    positions.push_back(mul2[0]);
    positions.push_back(mul2[1]);
    positions.push_back(mul2[2]);
    
    colors.push_back(255);
    colors.push_back(255);
    colors.push_back(255);
    colors.push_back(255);
    

    MultiplyMatrices(1, 4, 3, getUMatrix(u + 1), mul1, mul2);
    computeTangent(u+1, mul1);
    // glm::vec3 *square3 = getSquare (mul2, tangents.size() - 1);

    positions.push_back(mul2[0]);
    positions.push_back(mul2[1]);
    positions.push_back(mul2[2]);

    colors.push_back(255);
    colors.push_back(255);
    colors.push_back(255);
    colors.push_back(255);

    // cout << "X2 Y2 Z2 " << endl;
    // cout << mul2[0]<< " " << mul2[1]<< " " << mul2[2]<< " " << endl;
  }

}

void catMullSpline(){
  double * basis = getBasisMatrix(0.5);
  // vector <float> positions, colors;
  // makeVBO(basis, getControlMatrix(2), positions, colors);
  for (int i = 0; i < spline.numControlPoints - 3; i++){
    makeVBO(basis, getControlMatrix(i), positions, colors);
    // makeVBO(basis, getControlMatrix(i + 1), positions, colors);
  }
  computeUpVector();
  makeCubeVBO();
  // cout << "DONZO! " << endl;
  // numVertices = positions.size() / 3;

  cout << " NUM VERTICES for SPLINE : " << positions.size()/3 << endl;
  cout << " NUM Tangents of spline : " << tangents.size()/3 << endl;
  // cout << " NUM VERTICES for SPLINE : " << positions.size()/3 << endl;
  cout << " NUM up vectors : " << upVectors.size()/3 << endl;
}

void initGroundTexture(){
  vector<float> texPos, texUV;

  texPos.push_back(-256); texPos.push_back(-10); texPos.push_back(0); texUV.push_back (0); texUV.push_back (0); 
  texPos.push_back(0); texPos.push_back(-10); texPos.push_back(-256); texUV.push_back (1); texUV.push_back (0); 
  texPos.push_back(256); texPos.push_back(-10); texPos.push_back(0); texUV.push_back (1); texUV.push_back (1); 

  texPos.push_back(-256); texPos.push_back(-10); texPos.push_back(0); texUV.push_back (0); texUV.push_back (0);
  texPos.push_back(0); texPos.push_back(-10); texPos.push_back(256); texUV.push_back (0); texUV.push_back (1);
  texPos.push_back(256); texPos.push_back(-10); texPos.push_back(0); texUV.push_back (1); texUV.push_back (1); 

  glGenTextures(1, &texHandle);

  int code = initTexture("splines/sand.jpg", texHandle);
  if (code != 0){
    cout << "Error loading the texture image" << endl; 
    exit(EXIT_FAILURE);
  }

  int texNum = texPos.size()/3;


  vboVerticesTex = new VBO(texNum, 3, texPos.data(), GL_STATIC_DRAW); // 3 values per position
  VBO * vboUVs = new VBO(texNum, 2, texUV.data(), GL_STATIC_DRAW);
  // vboColorsTex = new VBO(texNum, 4, texCol.data(), GL_STATIC_DRAW); // 4 values per color
  vaoTex = new VAO();

  // Set up the relationship between the "position" shader variable and the VAO.
  // Important: any typo in the shader variable name will lead to malfunction.
  vaoTex->ConnectPipelineProgramAndVBOAndShaderVariable(texturePipeline, vboVerticesTex, "position");

  // Set up the relationship between the "color" shader variable and the VAO.
  // Important: any typo in the shader variable name will lead to malfunction.
  vaoTex->ConnectPipelineProgramAndVBOAndShaderVariable(texturePipeline, vboUVs, "texCoord");
}

void initScene(int argc, char *argv[])
{
  loadSpline(argv[1]);

  printf("Loaded spline with %d control point(s).\n", spline.numControlPoints);

  // Set the background color.
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black color.

  // Enable z-buffering (i.e., hidden surface removal using the z-buffer algorithm).
  glEnable(GL_DEPTH_TEST);

  // Create a pipeline program. This operation must be performed BEFORE we initialize any VAOs.
  // A pipeline program contains our shaders. Different pipeline programs may contain different shaders.
  // In this homework, we only have one set of shaders, and therefore, there is only one pipeline program.
  // In hw2, we will need to shade different objects with different shaders, and therefore, we will have
  // several pipeline programs (e.g., one for the rails, one for the ground/sky, etc.).
  pipelineProgram = new PipelineProgram(); // Load and set up the pipeline program, including its shaders.
  // Load and set up the pipeline program, including its shaders.
  if (pipelineProgram->BuildShadersFromFiles(shaderBasePath, "vertexShader.glsl", "fragmentShader.glsl") != 0)
  {
    cout << "Failed to build the pipeline program." << endl;
    throw 1;
  } 
  cout << "Successfully built the pipeline program." << endl;

  texturePipeline = new PipelineProgram ();
  if (texturePipeline->BuildShadersFromFiles(shaderBasePath, "texVertShader.glsl", "texFragShader.glsl") != 0)
  {
    cout << "Failed to build the  Texture pipeline program." << endl;
    throw 1;
  } 
    
  // Bind the pipeline program that we just created. 
  // The purpose of binding a pipeline program is to activate the shaders that it contains, i.e.,
  // any object rendered from that point on, will use those shaders.
  // When the application starts, no pipeline program is bound, which means that rendering is not set up.
  // So, at some point (such as below), we need to bind a pipeline program.
  // From that point on, exactly one pipeline program is bound at any moment of time.
  pipelineProgram->Bind();
  texturePipeline->Bind();

  std::cout << "GL error status is: " << glGetError() << std::endl;
  initGroundTexture();
  catMullSpline(); 

}



int main(int argc, char *argv[])
{
  
  if (argc < 2)
  {  
    printf ("Usage: %s <spline file>\n", argv[0]);
    exit(0);
  }

  // Load spline from the provided filename.

  cout << "Initializing GLUT..." << endl;
  glutInit(&argc,argv);

  cout << "Initializing OpenGL..." << endl;

  #ifdef __APPLE__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #else
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
  #endif

  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(0, 0);  
  glutCreateWindow(windowTitle);

  cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
  cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
  cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

  #ifdef __APPLE__
    // This is needed on recent Mac OS X versions to correctly display the window.
    glutReshapeWindow(windowWidth - 1, windowHeight - 1);
  #endif

  // Tells GLUT to use a particular display function to redraw.
  glutDisplayFunc(displayFunc);
  // Perform animation inside idleFunc.
  glutIdleFunc(idleFunc);
  // callback for mouse drags
  // glutMotionFunc(mouseMotionDragFunc);
  // callback for idle mouse movement
  // glutPassiveMotionFunc(mouseMotionFunc);
  // callback for mouse button changes
  // glutMouseFunc(mouseButtonFunc);
  // callback for resizing the window
  glutReshapeFunc(reshapeFunc);
  // callback for pressing the keys on the keyboard
  glutKeyboardFunc(keyboardFunc);

  // init glew
  #ifdef __APPLE__
    // nothing is needed on Apple
  #else
    // Windows, Linux
    GLint result = glewInit();
    if (result != GLEW_OK)
    {
      cout << "error: " << glewGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
    }
  #endif

  // Perform the initialization.
  initScene(argc, argv);

  // Sink forever into the GLUT loop.
  glutMainLoop();
}

