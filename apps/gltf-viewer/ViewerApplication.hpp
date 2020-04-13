#pragma once

#include "utils/GLFWHandle.hpp"
#include "utils/cameras.hpp"
#include "utils/filesystem.hpp"
#include "utils/shaders.hpp"

#include <tiny_gltf.h>

class ViewerApplication
{
public:
  ViewerApplication(const fs::path &appPath, uint32_t width, uint32_t height,
      const fs::path &gltfFile, const std::vector<float> &lookatArgs,
      const std::string &vertexShader, const std::string &fragmentShader,
      const fs::path &output);

  int run();

private:

  // Texture objects
  enum GBufferTextureType
  {
    GPosition = 0,
    GNormal,
    GAmbient,
    GDiffuse,
    GEmissive,
    GSpecular,
    GDepth, // On doit créer une texture de depth mais on écrit pas directement dedans dans le FS. OpenGL le fait pour nous (et l'utilise).
    GBufferTextureCount
  };

  GLuint m_GBufferTextures[GBufferTextureCount];
  const GLenum m_GBufferTextureFormat[GBufferTextureCount] = { GL_RGB32F, GL_RGB32F, GL_RGB32F, GL_RGB32F, GL_RGB32F, GL_RGB32F, GL_DEPTH_COMPONENT32F };
  const GLenum m_GBufferPixelFormat[GBufferTextureCount] = { GL_RGB, GL_RGB, GL_RGB, GL_RGB, GL_RGB, GL_RGB, GL_DEPTH_COMPONENT };
  const GLenum m_GBufferTextureAttachment[GBufferTextureCount] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_DEPTH_ATTACHMENT };

  GLuint m_FBO;

  // A range of indices in a vector containing Vertex Array Objects
  struct VaoRange
  {
    GLsizei begin; // Index of first element in vertexArrayObjects
    GLsizei count; // Number of elements in range
  };

  GLsizei m_nWindowWidth = 1280;
  GLsizei m_nWindowHeight = 720;

  const fs::path m_AppPath;
  const std::string m_AppName;
  const fs::path m_ShadersRootPath;

  fs::path m_gltfFilePath;
  std::string m_vertexShader = "geometryPass.vs.glsl";
  std::string m_fragmentShader = "geometryPass.fs.glsl";//"diffuse_directional_light.fs.glsl";//"normals.fs.glsl";
  std::string m_compileShader = "defered.cs.glsl";


  bool m_hasUserCamera = false;
  Camera m_userCamera;

  fs::path m_OutputPath;

  // Order is important here, see comment below
  const std::string m_ImGuiIniFilename;
  // Last to be initialized, first to be destroyed:
  GLFWHandle m_GLFWHandle{int(m_nWindowWidth), int(m_nWindowHeight),
      "glTF Viewer",
      m_OutputPath.empty()}; // show the window only if m_OutputPath is empty
  /*
    ! THE ORDER OF DECLARATION OF MEMBER VARIABLES IS IMPORTANT !
    - m_ImGuiIniFilename.c_str() will be used by ImGUI in ImGui::Shutdown, which
    will be called in destructor of m_GLFWHandle. So we must declare
    m_ImGuiIniFilename before m_GLFWHandle so that m_ImGuiIniFilename
    destructor is called after.
    - m_GLFWHandle must be declared before the creation of any object managing
    OpenGL resources (e.g. GLProgram, GLShader) because it is responsible for
    the creation of a GLFW windows and thus a GL context which must exists
    before most of OpenGL function calls.
  */

  // load the GLtf file from the model
  bool loadGltfFile(tinygltf::Model &model);

  // create the buffer Objects for OpenGL from the model
  std::vector<GLuint> createBufferObjects(const tinygltf::Model &model);

  // create the VAO for OpenGL from the model
  std::vector<GLuint> createVertexArrayObjects(
    const tinygltf::Model &model,
    const std::vector<GLuint> &bufferObjects,
    std::vector<VaoRange> &meshIndexToVaoRange
  );

  // create the Texture Objects from the model
  std::vector<GLuint> createTextureObjects(const tinygltf::Model &model) const;
};