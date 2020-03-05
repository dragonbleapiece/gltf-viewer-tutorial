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
#include "utils/gltf.hpp"
#include "utils/images.hpp"

// Each vertex attribute is identified by an index
// What vertex attribute we use, and what are their index is defined by the vertex shader
// we will use (more information later).
// position, normal and texcoord is pretty standard for 3D applications
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


  // LIGHTS

  const auto lightDirectionLocation =
      glGetUniformLocation(glslProgram.glId(), "uLightDirection");
  const auto lightIntensityLocation =
      glGetUniformLocation(glslProgram.glId(), "uLightIntensity");


  // TEXTURES

  const auto baseColorTextureLocation =
      glGetUniformLocation(glslProgram.glId(), "uBaseColorTexture");
  const auto baseColorFactorLocation =
      glGetUniformLocation(glslProgram.glId(), "uBaseColorFactor");

  // Materials

  const auto metallicFactorLocation =
      glGetUniformLocation(glslProgram.glId(), "uMetallicFactor");
  const auto roughnessFactorLocation =
      glGetUniformLocation(glslProgram.glId(), "uRoughnessFactor");
  const auto metallicRoughnessTextureLocation =
      glGetUniformLocation(glslProgram.glId(), "uMetallicRoughnessTexture");

  // LOADS MODEL

  tinygltf::Model model;
  // TODO Loading the glTF file
  if(!loadGltfFile(model)) {
    return -1;
  }

  // Bounding Box of the scene
  glm::vec3 bboxMin;
  glm::vec3 bboxMax;

  computeSceneBounds(
    model,
    bboxMin,
    bboxMax
  );

  glm::vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
  glm::vec3 bboxDiagonal = (bboxMax - bboxMin);

  float epsilon = 0.001f;
  // check if the scene is flat
  glm::vec3 up = glm::vec3(0, 1, 0);
  glm::vec3 eye = (bboxDiagonal.z < epsilon) ? bboxCenter + 2.f * glm::cross(bboxDiagonal, up) : bboxCenter + bboxDiagonal;

   // Build projection matrix
  auto maxDistance = glm::length(bboxDiagonal) > epsilon ? glm::length(bboxDiagonal) : 100.f;
  const auto projMatrix =
      glm::perspective(70.f, float(m_nWindowWidth) / m_nWindowHeight,
          0.001f * maxDistance, 1.5f * maxDistance);


  // TODO Implement a new CameraController model and use it instead. Propose the
  // choice from the GUI
  std::unique_ptr<CameraController> cameraController = std::make_unique<TrackballCameraController>(m_GLFWHandle.window(), 0.01f);
  if (m_hasUserCamera) {
    cameraController->setCamera(m_userCamera);
  } else {
    // TODO Use scene bounds to compute a better default camera
    cameraController->setCamera(
        Camera{eye, bboxCenter, up});
  }


  // Generate white texture
  GLuint whiteTexture;
  float white[] = {1., 1., 1., 1.};

  // Generate the texture object
  glGenTextures(1, &whiteTexture);

  glBindTexture(GL_TEXTURE_2D, whiteTexture); // Bind to target GL_TEXTURE_2D
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0,
          GL_RGBA, GL_FLOAT, white); // Set image data
  // Set sampling parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // Set wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);

  glBindTexture(GL_TEXTURE_2D, 0);

  // Creation of Texture Objects
  const std::vector<GLuint> textureObjects = createTextureObjects(model);

  // TODO Creation of Buffer Objects
  const std::vector<GLuint> bufferObjects = createBufferObjects(model);


  // TODO Creation of Vertex Array Objects
  std::vector<VaoRange> meshIndexToVaoRange;
  const std::vector<GLuint> vertexArrayObjects = createVertexArrayObjects(model, bufferObjects, meshIndexToVaoRange);

  // Setup OpenGL state for rendering
  glEnable(GL_DEPTH_TEST);
  glslProgram.use();

  // Light Variables
  glm::vec3 lightDirection(1., 1., 1.);
  glm::vec3 lightIntensity(1., 1., 1.);

  // Lambda function to bind material
  const auto bindMaterial = [&](const auto materialIndex) {
    // Material Binding

    if(materialIndex >= 0) {
      // only valid if materialIndex >= 0
      const auto &material = model.materials[materialIndex];
      const auto &pbrMetallicRoughness = material.pbrMetallicRoughness;
      
      // Base Color
      if(pbrMetallicRoughness.baseColorTexture.index >= 0) {
        // only valid if pbrMetallicRoughness.baseColorTexture.index >= 0:
        //const auto &texture = model.textures[pbrMetallicRoughness.baseColorTexture.index];
        const auto &texture = textureObjects[pbrMetallicRoughness.baseColorTexture.index];


        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        // By setting the uniform to 0, we tell OpenGL the texture is bound on tex unit 0:
        glUniform1i(baseColorTextureLocation, 0);
        glUniform4f(
            baseColorFactorLocation,
            (float)pbrMetallicRoughness.baseColorFactor[0],
            (float)pbrMetallicRoughness.baseColorFactor[1],
            (float)pbrMetallicRoughness.baseColorFactor[2],
            (float)pbrMetallicRoughness.baseColorFactor[3]
        );
      }
      if(pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
        const auto &texture = textureObjects[pbrMetallicRoughness.metallicRoughnessTexture.index];

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(metallicRoughnessTextureLocation, 1);
        glUniform1f(metallicFactorLocation, pbrMetallicRoughness.metallicFactor);
        glUniform1f(roughnessFactorLocation, pbrMetallicRoughness.roughnessFactor);
      }

    } else {

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, whiteTexture);
      // By setting the uniform to 0, we tell OpenGL the texture is bound on tex unit 0:
      glUniform1i(baseColorTextureLocation, 0);
      glUniform4f(baseColorFactorLocation, 1., 1., 1., 1.);
    }

  };


  // Lambda function to draw the scene
  const auto drawScene = [&](const Camera &camera) {
    glViewport(0, 0, m_nWindowWidth, m_nWindowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto viewMatrix = camera.getViewMatrix();

    glm::vec3 ligthDirInViewSpace(glm::normalize(viewMatrix * glm::vec4(lightDirection, 0.))); 

    if(lightDirectionLocation >= 0 ) {
      glUniform3fv(lightDirectionLocation, 1, glm::value_ptr(ligthDirInViewSpace));
    }

    if(lightIntensityLocation >= 0) {
      glUniform3fv(lightIntensityLocation, 1, glm::value_ptr(lightIntensity));
    }

    // The recursive function that should draw a node
    // We use a std::function because a simple lambda cannot be recursive
    const std::function<void(int, const glm::mat4 &)> drawNode =
        [&](int nodeIdx, const glm::mat4 &parentMatrix) {
          // TODO The drawNode function
          const auto &node = model.nodes[nodeIdx];

          glm::mat4 modelMatrix = getLocalToWorldMatrix(node, parentMatrix);

          if(node.mesh >= 0) {
            const glm::mat4 modelViewMatrix = viewMatrix * modelMatrix;
            const glm::mat4 modelViewProjMatrix = projMatrix * modelViewMatrix;
            const glm::mat4 normalMatrix = glm::transpose(glm::inverse(modelViewMatrix));

            glUniformMatrix4fv(modelViewMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelViewMatrix));
            glUniformMatrix4fv(modelViewProjMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelViewProjMatrix));
            glUniformMatrix4fv(normalMatrixLocation, 1, GL_FALSE, glm::value_ptr(normalMatrix));

            const auto &mesh = model.meshes[node.mesh];
            const auto &vaoRange = meshIndexToVaoRange[node.mesh];

            for(size_t primitiveIdx = 0; primitiveIdx < mesh.primitives.size(); ++primitiveIdx) {
              const auto &primitive = mesh.primitives[primitiveIdx];
              bindMaterial(primitive.material);
              
              const GLuint vao = vertexArrayObjects[vaoRange.begin + primitiveIdx];

              glBindVertexArray(vao);

              if(primitive.indices >= 0) {
                const auto &accessor = model.accessors[primitive.indices];
                const auto &bufferView = model.bufferViews[accessor.bufferView];
                const auto byteOffset = bufferView.byteOffset + accessor.byteOffset;

                glDrawElements(primitive.mode, GLsizei(accessor.count), accessor.componentType,(const GLvoid*)byteOffset);
              } else {
                const auto accessorIdx = (*begin(primitive.attributes)).second;
                const auto &accessor = model.accessors[accessorIdx];
                glDrawArrays(primitive.mode, 0, GLsizei(accessor.count));
              }
            }
            glBindVertexArray(0);
          }

          // Draw children
          for (const auto childNodeIdx : node.children) {
            drawNode(childNodeIdx, modelMatrix);
          }
        };

    // Draw the scene referenced by gltf file
    if (model.defaultScene >= 0) {
      // TODO Draw all nodes
      for(const auto nodeIdx : model.scenes[model.defaultScene].nodes) {
        drawNode(nodeIdx, glm::mat4(1));
      }
    }
  };

  // Render to image output
  if(!m_OutputPath.empty()) {
    const int numComponents = 3;
    std::vector<unsigned char> pixels(m_nWindowWidth * m_nWindowHeight * numComponents);
    renderToImage(m_nWindowWidth, m_nWindowHeight, numComponents, pixels.data(), [&]() {
      drawScene(cameraController->getCamera());
    });
    flipImageYAxis(m_nWindowWidth, m_nWindowHeight, numComponents, pixels.data());
    const auto strPath = m_OutputPath.string();
    stbi_write_png(
      strPath.c_str(),
      m_nWindowWidth,
      m_nWindowHeight,
      numComponents,
      pixels.data(),
      0
    );

    return 0;
  }

  // Loop until the user closes the window
  for (auto iterationCount = 0u; !m_GLFWHandle.shouldClose();
       ++iterationCount) {
    const auto seconds = glfwGetTime();

    const auto camera = cameraController->getCamera();
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

        // Radio buttons to switch camera type
        static int cameraControllerType = 0;
        if (ImGui::RadioButton("Trackball", &cameraControllerType, 0)) {
          cameraController = std::make_unique<TrackballCameraController>(m_GLFWHandle.window(), 0.01f);
          cameraController->setCamera(camera);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("FirstPerson", &cameraControllerType, 1)) {
          cameraController = std::make_unique<FirstPersonCameraController>(m_GLFWHandle.window(), 5.f * maxDistance);
          cameraController->setCamera(camera);
        }

        if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
          static float theta = 0.0f;
          static float phi = 0.0f;
          static bool lightFromCamera = true;
          ImGui::Checkbox("Light from camera", &lightFromCamera);
          if (lightFromCamera) {
            lightDirection = -camera.front();
          } else {
            if (ImGui::SliderFloat("Theta", &theta, 0.f, glm::pi<float>()) ||
                ImGui::SliderFloat("Phi", &phi, 0, 2.f * glm::pi<float>())) {
              lightDirection = glm::vec3(
                glm::sin(theta) * glm::cos(phi),
                glm::cos(theta),
                glm::sin(theta) * glm::sin(phi)
              );
            }
          }

          static glm::vec3 lightColor(1.f, 1.f, 1.f);
          static float lightIntensityFactor = 1.f;
          if (ImGui::ColorEdit3("Light Color", (float *)&lightColor) ||
              ImGui::SliderFloat("Ligth Intensity", &lightIntensityFactor, 0.f, 10.f)) {
            lightIntensity = lightColor * lightIntensityFactor;
          }
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
      cameraController->update(float(ellapsedTime));
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
    std::cerr << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!ret) {
    std::cerr << "Failed to parse glTF file" << std::endl;
    return false;
  }

  return ret;
}

std::vector<GLuint> ViewerApplication::createBufferObjects(const tinygltf::Model &model) {
  std::vector<tinygltf::Buffer> buffers(model.buffers);

  std::vector<GLuint> bufferObjects(buffers.size(), 0); // Assuming buffers is a std::vector of Buffer
  glGenBuffers(GLsizei(bufferObjects.size()), bufferObjects.data()); // Ask opengl to reserve an identifier for our buffer object and store it in bufferObject.
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
    const auto &mesh = model.meshes[meshIdx];

    vertexArrayObjects.resize(vaoOffset + mesh.primitives.size());
    meshIndexToVaoRange.push_back(VaoRange{GLsizei(vaoOffset), GLsizei(mesh.primitives.size())});  // Will be used during rendering

    glGenVertexArrays(GLsizei(mesh.primitives.size()), &vertexArrayObjects[vaoOffset]);

    for(size_t primitiveIdx = 0; primitiveIdx < mesh.primitives.size(); ++primitiveIdx) {
      glBindVertexArray(vertexArrayObjects[vaoOffset + primitiveIdx]);

      const auto &primitive = mesh.primitives[primitiveIdx];

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
          glEnableVertexAttribArray(attrIdx);
          // TODO Bind the buffer object to GL_ARRAY_BUFFER
          glBindBuffer(GL_ARRAY_BUFFER, bufferObject);

          const auto byteOffset = bufferView.byteOffset + accessor.byteOffset;

          // TODO Call glVertexAttribPointer with the correct arguments. 
          glVertexAttribPointer(
            attrIdx,
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

std::vector<GLuint> ViewerApplication::createTextureObjects(const tinygltf::Model &model) const {

  tinygltf::Sampler defaultSampler;
  defaultSampler.minFilter = GL_LINEAR;
  defaultSampler.magFilter = GL_LINEAR;
  defaultSampler.wrapS = GL_REPEAT;
  defaultSampler.wrapT = GL_REPEAT;
  defaultSampler.wrapR = GL_REPEAT;

  std::vector<GLuint> textureArrayObjects;

  for(size_t i = 0; i < model.textures.size(); ++i) {

    GLuint texObject;
    // Generate the texture object
    glGenTextures(1, &texObject);


    glBindTexture(GL_TEXTURE_2D, texObject); // Bind to target GL_TEXTURE_2D

    // Assume a texture object has been created and bound to GL_TEXTURE_2D
    const auto &texture = model.textures[i]; // get i-th texture
    assert(texture.source >= 0); // ensure a source image is present
    const auto &image = model.images[texture.source]; // get the image
    const auto &sampler =
        texture.sampler >= 0 ? model.samplers[texture.sampler] : defaultSampler;


    // fill the texture object with the data from the image
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0,
            GL_RGBA, image.pixel_type, image.image.data());

    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
      sampler.minFilter != -1 ? sampler.minFilter : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
      sampler.magFilter != -1 ? sampler.magFilter : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, sampler.wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, sampler.wrapT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, sampler.wrapR);

    if (sampler.minFilter == GL_NEAREST_MIPMAP_NEAREST ||
      sampler.minFilter == GL_NEAREST_MIPMAP_LINEAR ||
      sampler.minFilter == GL_LINEAR_MIPMAP_NEAREST ||
      sampler.minFilter == GL_LINEAR_MIPMAP_LINEAR) {
      glGenerateMipmap(GL_TEXTURE_2D);
    }

    textureArrayObjects.push_back(texObject);
  }

  glBindTexture(GL_TEXTURE_2D, 0);

  return textureArrayObjects;
}