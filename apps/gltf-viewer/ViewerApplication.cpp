#include "ViewerApplication.hpp"

#include <iostream>
#include <numeric>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/io.hpp>

#include "utils/cameras.hpp"

#include <stb_image_write.h>
#include <tiny_gltf.h>

// Each vertex attribute is identified by an index
// What vertex attribute we use, and what are their index is defined by the vertex shader
// we will use (more information later).
// position, normal and texcoord is pretty standard for 3D applications
const GLuint VERTEX_ATTRIB_POSITION_IDX = 0;
const GLuint VERTEX_ATTRIB_NORMAL_IDX = 1;
const GLuint VERTEX_ATTRIB_TEXCOORD0_IDX = 2;
const std::vector<std::string> ATTRIBUTES = {"POSITION", "NORMAL", "TEXCOORD_0"};

void keyCallback(
    GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
    glfwSetWindowShouldClose(window, 1);
  }
}

int ViewerApplication::run()
{
  // Loader shaders
  const auto glslProgram =
      compileProgram({m_ShadersRootPath / m_AppName / m_vertexShader,
          m_ShadersRootPath / m_AppName / m_fragmentShader});

  const auto modelViewProjMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uModelViewProjMatrix");
  const auto modelViewMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uModelViewMatrix");
  const auto normalMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uNormalMatrix");

  // Build projection matrix
  auto maxDistance = 500.f; // TODO use scene bounds instead to compute this
  maxDistance = maxDistance > 0.f ? maxDistance : 100.f;
  const auto projMatrix =
      glm::perspective(70.f, float(m_nWindowWidth) / m_nWindowHeight,
          0.001f * maxDistance, 1.5f * maxDistance);

  // TODO Implement a new CameraController model and use it instead. Propose the
  // choice from the GUI
  FirstPersonCameraController cameraController{
      m_GLFWHandle.window(), 0.5f * maxDistance};
  if (m_hasUserCamera) {
    cameraController.setCamera(m_userCamera);
  } else {
    // TODO Use scene bounds to compute a better default camera
    cameraController.setCamera(
        Camera{glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0)});
  }

  tinygltf::Model model;
  // TODO Loading the glTF file
  loadGltfFile(model);

  // TODO Creation of Buffer Objects
  std::vector<GLuint> bufferObjects = createBufferObjects(model);


  // TODO Creation of Vertex Array Objects
  std::vector<VaoRange> meshIndexToVaoRange;
  std::vector<GLuint> vertexArrayObjects = createVertexArrayObjects(model, bufferObjects, meshIndexToVaoRange);

  // Setup OpenGL state for rendering
  glEnable(GL_DEPTH_TEST);
  glslProgram.use();

  // Lambda function to draw the scene
  const auto drawScene = [&](const Camera &camera) {
    glViewport(0, 0, m_nWindowWidth, m_nWindowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto viewMatrix = camera.getViewMatrix();

    // The recursive function that should draw a node
    // We use a std::function because a simple lambda cannot be recursive
    const std::function<void(int, const glm::mat4 &)> drawNode =
        [&](int nodeIdx, const glm::mat4 &parentMatrix) {
          // TODO The drawNode function
        };

    // Draw the scene referenced by gltf file
    if (model.defaultScene >= 0) {
      // TODO Draw all nodes
      for(size_t nodeIdx = 0; nodeIdx < model.scenes[model.defaultScene].nodes.size(); ++nodeIdx) {
        drawNode(nodeIdx, glm::mat4(1));
      }
    }
  };

  // Loop until the user closes the window
  for (auto iterationCount = 0u; !m_GLFWHandle.shouldClose();
       ++iterationCount) {
    const auto seconds = glfwGetTime();

    const auto camera = cameraController.getCamera();
    drawScene(camera);

    // GUI code:
    imguiNewFrame();

    {
      ImGui::Begin("GUI");
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
          1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("eye: %.3f %.3f %.3f", camera.eye().x, camera.eye().y,
            camera.eye().z);
        ImGui::Text("center: %.3f %.3f %.3f", camera.center().x,
            camera.center().y, camera.center().z);
        ImGui::Text(
            "up: %.3f %.3f %.3f", camera.up().x, camera.up().y, camera.up().z);

        ImGui::Text("front: %.3f %.3f %.3f", camera.front().x, camera.front().y,
            camera.front().z);
        ImGui::Text("left: %.3f %.3f %.3f", camera.left().x, camera.left().y,
            camera.left().z);

        if (ImGui::Button("CLI camera args to clipboard")) {
          std::stringstream ss;
          ss << "--lookat " << camera.eye().x << "," << camera.eye().y << ","
             << camera.eye().z << "," << camera.center().x << ","
             << camera.center().y << "," << camera.center().z << ","
             << camera.up().x << "," << camera.up().y << "," << camera.up().z;
          const auto str = ss.str();
          glfwSetClipboardString(m_GLFWHandle.window(), str.c_str());
        }
      }
      ImGui::End();
    }

    imguiRenderFrame();

    glfwPollEvents(); // Poll for and process events

    auto ellapsedTime = glfwGetTime() - seconds;
    auto guiHasFocus =
        ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;
    if (!guiHasFocus) {
      cameraController.update(float(ellapsedTime));
    }

    m_GLFWHandle.swapBuffers(); // Swap front and back buffers
  }

  // TODO clean up allocated GL data

  return 0;
}

ViewerApplication::ViewerApplication(const fs::path &appPath, uint32_t width,
    uint32_t height, const fs::path &gltfFile,
    const std::vector<float> &lookatArgs, const std::string &vertexShader,
    const std::string &fragmentShader, const fs::path &output) :
    m_nWindowWidth(width),
    m_nWindowHeight(height),
    m_AppPath{appPath},
    m_AppName{m_AppPath.stem().string()},
    m_ImGuiIniFilename{m_AppName + ".imgui.ini"},
    m_ShadersRootPath{m_AppPath.parent_path() / "shaders"},
    m_gltfFilePath{gltfFile},
    m_OutputPath{output}
{
  if (!lookatArgs.empty()) {
    m_hasUserCamera = true;
    m_userCamera =
        Camera{glm::vec3(lookatArgs[0], lookatArgs[1], lookatArgs[2]),
            glm::vec3(lookatArgs[3], lookatArgs[4], lookatArgs[5]),
            glm::vec3(lookatArgs[6], lookatArgs[7], lookatArgs[8])};
  }

  if (!vertexShader.empty()) {
    m_vertexShader = vertexShader;
  }

  if (!fragmentShader.empty()) {
    m_fragmentShader = fragmentShader;
  }

  ImGui::GetIO().IniFilename =
      m_ImGuiIniFilename.c_str(); // At exit, ImGUI will store its windows
                                  // positions in this file

  glfwSetKeyCallback(m_GLFWHandle.window(), keyCallback);

  printGLVersion();
}

bool ViewerApplication::loadGltfFile(tinygltf::Model &model) {
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, m_gltfFilePath.string());
  //bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, m_gltfFilePath.string()); // for binary glTF(.glb)

  if (!warn.empty()) {
    std::printf("Warn: %s\n", warn.c_str());
  }

  if (!err.empty()) {
    std::printf("Err: %s\n", err.c_str());
  }

  if (!ret) {
    std::printf("Failed to parse glTF\n");
    return -1;
  }

  return ret;
}

std::vector<GLuint> ViewerApplication::createBufferObjects(const tinygltf::Model &model) {
  std::vector<tinygltf::Buffer> buffers(model.buffers);

  std::vector<GLuint> bufferObjects(buffers.size(), 0); // Assuming buffers is a std::vector of Buffer
  glGenBuffers(bufferObjects.size(), bufferObjects.data()); // Ask opengl to reserve an identifier for our buffer object and store it in bufferObject.
  for (size_t i = 0; i < buffers.size(); ++i) {
    glBindBuffer(GL_ARRAY_BUFFER, bufferObjects[i]);
    glBufferStorage(GL_ARRAY_BUFFER, buffers[i].data.size(), // Assume a Buffer has a data member variable of type std::vector
        buffers[i].data.data(), 0);
  }
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Cleanup the binding point after the loop only

  return bufferObjects;
}

std::vector<GLuint> ViewerApplication::createVertexArrayObjects(
  const tinygltf::Model &model,
  const std::vector<GLuint> &bufferObjects,
  std::vector<VaoRange> &meshIndexToVaoRange
) {
  std::vector<GLuint> vertexArrayObjects;

  for (size_t meshIdx = 0; meshIdx < model.meshes.size(); ++meshIdx) {
    const auto vaoOffset = vertexArrayObjects.size();
    tinygltf::Mesh mesh = model.meshes[meshIdx];

    vertexArrayObjects.resize(vaoOffset + mesh.primitives.size());
    meshIndexToVaoRange.push_back(VaoRange{(GLsizei)vaoOffset, (GLsizei)mesh.primitives.size()});  // Will be used during rendering

    glGenVertexArrays(mesh.primitives.size(), &vertexArrayObjects[vaoOffset]);

    for(size_t primitiveIdx = 0; primitiveIdx < mesh.primitives.size(); ++primitiveIdx) {
      glBindVertexArray(vertexArrayObjects[vaoOffset + primitiveIdx]);

      tinygltf::Primitive primitive = mesh.primitives[primitiveIdx];

      for(size_t attrIdx = 0; attrIdx < ATTRIBUTES.size(); ++attrIdx) {
        const auto iterator = primitive.attributes.find(ATTRIBUTES[attrIdx]);
        if (iterator != end(primitive.attributes)) { // If "POSITION" has been found in the map
          // (*iterator).first is the key "POSITION", (*iterator).second is the value, ie. the index of the accessor for this attribute
          const auto accessorIdx = (*iterator).second;
          const auto &accessor = model.accessors[accessorIdx];
          const auto &bufferView = model.bufferViews[accessor.bufferView];
          const auto bufferIdx = bufferView.buffer;

          const auto bufferObject = bufferObjects[bufferIdx];

          // TODO Enable the vertex attrib array corresponding to POSITION with glEnableVertexAttribArray (you need to use VERTEX_ATTRIB_POSITION_IDX which is defined at the top of the file)
          glEnableVertexAttribArray(VERTEX_ATTRIB_POSITION_IDX);
          // TODO Bind the buffer object to GL_ARRAY_BUFFER
          glBindBuffer(GL_ARRAY_BUFFER, bufferObject);

          const auto byteOffset = bufferView.byteOffset + accessor.byteOffset;

          // TODO Call glVertexAttribPointer with the correct arguments. 
          glVertexAttribPointer(
            VERTEX_ATTRIB_POSITION_IDX,
            accessor.type,
            accessor.componentType,
            GL_FALSE,
            bufferView.byteStride,
            (const GLvoid*)byteOffset
          );
          // Remember size is obtained with accessor.type, type is obtained with accessor.componentType. 
          // The stride is obtained in the bufferView, normalized is always GL_FALSE, and pointer is the byteOffset (don't forget the cast).
        }
      }

      if(primitive.indices >= 0) {
        const auto &accessor = model.accessors[primitive.indices];
        const auto &bufferView = model.bufferViews[accessor.bufferView];
        const auto bufferIdx = bufferView.buffer;
        const auto bufferObject = bufferObjects[bufferIdx];

        // Tell OpenGL we use an index buffer for this primitive:
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject);
      }

      // End the description of our vertex array object:
      glBindVertexArray(0);
    }
  }

  return vertexArrayObjects;
}