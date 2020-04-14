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

void printErrorCode() {
  GLenum error;
  bool hasError = false;
  while((error = glGetError()) != GL_NO_ERROR) {
    std::cout << "error code : " << error << std::endl; 
    hasError = true;
  }

  assert(!hasError);
}


// https://zestedesavoir.com/tutoriels/1554/introduction-aux-compute-shaders/
// check capabilities of GPU
void printWorkGroupsCapabilities() {
  int workgroup_count[3];
  int workgroup_size[3];
  int workgroup_invocations;

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &workgroup_count[0]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &workgroup_count[1]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &workgroup_count[2]);

  printf ("Taille maximale des workgroups:\n\tx:%u\n\ty:%u\n\tz:%u\n",
  workgroup_size[0], workgroup_size[1], workgroup_size[2]);

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &workgroup_size[0]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &workgroup_size[1]);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &workgroup_size[2]);

  printf ("Nombre maximal d'invocation locale:\n\tx:%u\n\ty:%u\n\tz:%u\n",
  workgroup_size[0], workgroup_size[1], workgroup_size[2]);

  glGetIntegerv (GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &workgroup_invocations);
  printf ("Nombre maximum d'invocation de workgroups:\n\t%u\n", workgroup_invocations);
}

int ViewerApplication::run()
{

  /// FOR DEFERRED RENDERING

  // Generate FBO
  glGenFramebuffers(1, &m_FBO);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_FBO);

  // Create texture objects and Bind textures to FBO
  glGenTextures(GBufferTextureCount, m_GBufferTextures);
  for(GLuint i = 0; i < GBufferTextureCount; ++i) {
    glBindTexture(GL_TEXTURE_2D, m_GBufferTextures[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, m_GBufferTextureFormat[i], m_nWindowWidth, m_nWindowHeight, 0, m_GBufferPixelFormat[i], GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, m_GBufferTextureAttachment[i], GL_TEXTURE_2D, m_GBufferTextures[i], 0);
  }

  glDrawBuffers(6, m_GBufferTextureAttachment);
  assert(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

  printWorkGroupsCapabilities();

  /// FOR SHADOW MAPPING

  // Generate FBO
  glGenFramebuffers(1, &m_directionalSMFBO);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_directionalSMFBO);

  // Generate Texture
  glGenTextures(1, &m_directionalSMTexture);
  glBindTexture(GL_TEXTURE_2D, m_directionalSMTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, m_nDirectionalSMResolution, m_nDirectionalSMResolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_directionalSMTexture, 0);

  assert(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

  // Generate sampler
  glGenSamplers(1, &m_directionalSMSampler);
  glSamplerParameteri(m_directionalSMSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glSamplerParameteri(m_directionalSMSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glSamplerParameteri(m_directionalSMSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glSamplerParameteri(m_directionalSMSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);


  // Loader shaders
  const auto glslProgram =
      compileProgram({m_ShadersRootPath / m_AppName / m_vertexShader,
          m_ShadersRootPath / m_AppName / m_fragmentShader});

  const auto compProgram =
      compileProgram({m_ShadersRootPath / m_AppName / m_computeShader});

  const auto directionalSMProgram =
      compileProgram({m_ShadersRootPath / m_AppName / m_vertexSMShader,
          m_ShadersRootPath / m_AppName / m_fragmentSMShader});

  // MATRICES

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
  const auto emissiveTextureLocation =
      glGetUniformLocation(glslProgram.glId(), "uEmissiveTexture");
  const auto emissiveFactorLocation =
      glGetUniformLocation(glslProgram.glId(), "uEmissiveFactor");


  // Compute Shader Uniforms

  const GLint cTexturesLocation[6] = {
    glGetUniformLocation(compProgram.glId(), "uPositionTexture"),
    glGetUniformLocation(compProgram.glId(), "uNormalTexture"),
    glGetUniformLocation(compProgram.glId(), "uAmbientTexture"),
    glGetUniformLocation(compProgram.glId(), "uDiffuseTexture"),
    glGetUniformLocation(compProgram.glId(), "uEmissiveTexture"),
    glGetUniformLocation(compProgram.glId(), "uSpecularTexture")
  };

  const auto cDirLightViewProjMatrixLocation =
      glGetUniformLocation(compProgram.glId(), "uDirLightViewProjMatrix");
  const auto cDirLightShadowMapBiasLocation =
      glGetUniformLocation(compProgram.glId(), "uDirLightShadowMapBias");
  const auto cDirLightShadowMapLocation =
      glGetUniformLocation(compProgram.glId(), "uDirLightShadowMap");

  // For depthMap display
  const auto cDepthDisplayLocation =
      glGetUniformLocation(compProgram.glId(), "uDisplayDepth");

  // ShadowMap Shader Uniforms
  const auto dirLightViewProjMatrixLocation = 
      glGetUniformLocation(directionalSMProgram.glId(), "uDirLightViewProjMatrix");
  const auto modelMatrixLocation = 
      glGetUniformLocation(directionalSMProgram.glId(), "uModelMatrix");

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


  const float cameraSpeed = 1.f;

  // Light Variables
  glm::vec3 lightDirection(1., 1., 1.);
  glm::vec3 lightIntensity(1., 1., 1.);

  // Lights calculations
  static const auto computeDirectionVectorUp = [](float phiRadians, float thetaRadians)
  {
      const auto cosPhi = glm::cos(phiRadians);
      const auto sinPhi = glm::sin(phiRadians);
      const auto cosTheta = glm::cos(thetaRadians);
      return -glm::normalize(glm::vec3(sinPhi * cosTheta, -glm::sin(thetaRadians), cosPhi * cosTheta));
  };

  const float sceneRadius = glm::length(bboxDiagonal) * 0.5f;

  auto dirLightUpVector = up;
  auto dirLightViewMatrix = glm::lookAt(eye + lightDirection * sceneRadius, eye, up);
  const auto dirLightProjMatrix = glm::ortho(-sceneRadius, sceneRadius, -sceneRadius, sceneRadius, 0.01f * sceneRadius, 2.f * sceneRadius);

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

  // variable for rendering mode
  GBufferTextureType renderMode = GPosition;

  // variable for shadow map recalculation
  bool directionalSMDirty = true;

  // variable for shadow map bias
  float shadowMapBias = 0.f;

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

  // Lambda function to bind material
  const auto bindMaterial = [&](const auto materialIndex) {
    // Material Binding

    if(materialIndex >= 0) {
      // only valid if materialIndex >= 0
      const auto &material = model.materials[materialIndex];
      const auto &pbrMetallicRoughness = material.pbrMetallicRoughness;
      const auto &emissiveTexture = material.emissiveTexture;
      
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

      // Roughness Texture
      if(pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
        const auto &texture = textureObjects[pbrMetallicRoughness.metallicRoughnessTexture.index];

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(metallicRoughnessTextureLocation, 1);
        glUniform1f(metallicFactorLocation, pbrMetallicRoughness.metallicFactor);
        glUniform1f(roughnessFactorLocation, pbrMetallicRoughness.roughnessFactor);
      }

      // EmissiveTexture
      if (emissiveTexture.index >= 0) {
        const auto &texture = textureObjects[emissiveTexture.index];
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(emissiveTextureLocation, 2);
        glUniform3f(
          emissiveFactorLocation, 
          (float) material.emissiveFactor[0],
          (float) material.emissiveFactor[1],
          (float) material.emissiveFactor[2]
        );
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
    glslProgram.use();
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

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_FBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw the scene referenced by gltf file
    if (model.defaultScene >= 0) {
      // TODO Draw all nodes
      for(const auto nodeIdx : model.scenes[model.defaultScene].nodes) {
        drawNode(nodeIdx, glm::mat4(1));
      }
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  };

  const auto drawShadowMap = [&]() {
    directionalSMProgram.use();
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_directionalSMFBO);
    glViewport(0, 0, m_nDirectionalSMResolution, m_nDirectionalSMResolution);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(dirLightViewProjMatrixLocation, 1, GL_FALSE, glm::value_ptr(dirLightProjMatrix * dirLightViewMatrix));

    // The recursive function that should draw a node
    // We use a std::function because a simple lambda cannot be recursive
    const std::function<void(int, const glm::mat4 &)> drawNode =
        [&](int nodeIdx, const glm::mat4 &parentMatrix) {
          // TODO The drawNode function
          const auto &node = model.nodes[nodeIdx];

          glm::mat4 modelMatrix = getLocalToWorldMatrix(node, parentMatrix);

          if(node.mesh >= 0) {
            
            glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));

            const auto &mesh = model.meshes[node.mesh];
            const auto &vaoRange = meshIndexToVaoRange[node.mesh];

            for(size_t primitiveIdx = 0; primitiveIdx < mesh.primitives.size(); ++primitiveIdx) {
              const auto &primitive = mesh.primitives[primitiveIdx];
              
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

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  };

  // For Compute shader result output

  GLuint fboResult;
  GLuint texResult;

  glGenTextures(1, &texResult);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texResult);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_nWindowWidth, m_nWindowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
  glBindImageTexture(0, texResult, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

  glGenFramebuffers(1, &fboResult);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fboResult);
  glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texResult, 0);
  assert(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

  const auto computeScene = [&](const Camera &camera) {
    
    if(renderMode < GBufferTextureCount - 1) { // If only one texture is rendered
      glBindFramebuffer(GL_READ_FRAMEBUFFER, m_FBO);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

      glReadBuffer(GL_COLOR_ATTACHMENT0 + renderMode);
      glBlitFramebuffer(0, 0, m_nWindowWidth, m_nWindowHeight, 0, 0, m_nWindowWidth, m_nWindowHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    } else if(renderMode == GDepth) { // For Depth Map
      compProgram.use();
      
      glBindFramebuffer(GL_READ_FRAMEBUFFER, fboResult);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

      glUniform1i(cDepthDisplayLocation, 1);
      glActiveTexture(GL_TEXTURE0);
      //glBindTexture(GL_TEXTURE_2D, m_directionalSMTexture);
      //glBindSampler(m_directionalSMTexture, m_directionalSMSampler);
      glBindTexture(GL_TEXTURE_2D, m_GBufferTextures[GDepth]);
      glUniform1i(cDirLightShadowMapLocation, 0);

      glDispatchCompute(m_nWindowWidth, m_nWindowHeight, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
      glBlitFramebuffer(0, 0, m_nWindowWidth, m_nWindowHeight, 0, 0, m_nWindowWidth, m_nWindowHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);

      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    } else { // Else, Mix of textures so call the compute shader
      compProgram.use();
      
      glBindFramebuffer(GL_READ_FRAMEBUFFER, fboResult);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

      glUniform1i(cDepthDisplayLocation, 0);

      for(GLuint i = 0; i < GBufferTextureCount - 1; ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, m_GBufferTextures[i]);
        glUniform1i(cTexturesLocation[i], i);
      }

      glActiveTexture(GL_TEXTURE0 + GBufferTextureCount - 1);
      glBindTexture(GL_TEXTURE_2D, m_directionalSMTexture);
      glBindSampler(m_directionalSMTexture, m_directionalSMSampler);
      glUniform1i(cDirLightShadowMapLocation, GBufferTextureCount - 1);

      //const auto rcpViewMatrix = glm::inverse(camera.getViewMatrix()); // Inverse de la view matrix de la caméra
      glUniformMatrix4fv(cDirLightViewProjMatrixLocation, 1, GL_FALSE, glm::value_ptr(dirLightProjMatrix * dirLightViewMatrix));

      glUniform1f(cDirLightShadowMapBiasLocation, shadowMapBias);


      glDispatchCompute(m_nWindowWidth, m_nWindowHeight, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
      glBlitFramebuffer(0, 0, m_nWindowWidth, m_nWindowHeight, 0, 0, m_nWindowWidth, m_nWindowHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);

      glActiveTexture(GL_TEXTURE0);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    }

    printErrorCode();
  };

  // Render to image output
  if(!m_OutputPath.empty()) {
    const int numComponents = 3;
    std::vector<unsigned char> pixels(m_nWindowWidth * m_nWindowHeight * numComponents);
    renderToImage(m_nWindowWidth, m_nWindowHeight, numComponents, pixels.data(), [&]() {
      const auto camera = cameraController->getCamera();
      drawShadowMap();
      drawScene(camera);
      computeScene(camera);
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

    // shadow mapping
    if(directionalSMDirty) {

      drawShadowMap();
      directionalSMDirty = false;
    }

    // rendering
    drawScene(camera);
    computeScene(camera);

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
          cameraController = std::make_unique<TrackballCameraController>(m_GLFWHandle.window(), cameraSpeed);
          cameraController->setCamera(camera);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("FirstPerson", &cameraControllerType, 1)) {
          cameraController = std::make_unique<FirstPersonCameraController>(m_GLFWHandle.window(), cameraSpeed * maxDistance);
          cameraController->setCamera(camera);
        }

        if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
          static float theta = 0.0f;
          static float phi = 0.0f;
          static bool lightFromCamera = true;
          ImGui::Checkbox("Light from camera", &lightFromCamera);
          auto temp = lightDirection; // check if lightDirection has changed
          if (lightFromCamera) {
            lightDirection = -camera.front();
            dirLightUpVector = camera.up();
            dirLightViewMatrix = glm::lookAt(camera.eye() + lightDirection * sceneRadius, camera.eye(), camera.up());
          } else {
            if (ImGui::SliderFloat("Theta", &theta, 0.f, glm::pi<float>()) ||
                ImGui::SliderFloat("Phi", &phi, 0, 2.f * glm::pi<float>())) {
              lightDirection = glm::vec3(
                glm::sin(theta) * glm::cos(phi),
                glm::cos(theta),
                glm::sin(theta) * glm::sin(phi)
              );
              dirLightUpVector = computeDirectionVectorUp(phi, theta);
              dirLightViewMatrix = glm::lookAt(bboxCenter + lightDirection * sceneRadius, bboxCenter, dirLightUpVector); // Will not work if lightDirection is colinear to lightUpVector
            }
          }

          static glm::vec3 lightColor(1.f, 1.f, 1.f);
          static float lightIntensityFactor = 1.f;
          if (ImGui::ColorEdit3("Light Color", (float *)&lightColor) ||
              ImGui::SliderFloat("Ligth Intensity", &lightIntensityFactor, 0.f, 10.f)) {
            lightIntensity = lightColor * lightIntensityFactor;
          }

          if(temp != lightDirection) {
            directionalSMDirty = true;
          }
        }

        if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
          // Radio buttons to switch render
          static int renderType = 0;
          if (ImGui::RadioButton("Position", &renderType, 0)) {
            renderMode = GPosition;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("Normal", &renderType, 1)) {
            renderMode = GNormal;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("Ambient", &renderType, 2)) {
            renderMode = GAmbient;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("Diffuse", &renderType, 3)) {
            renderMode = GDiffuse;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("Emissive", &renderType, 4)) {
            renderMode = GEmissive;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("Specular", &renderType, 5)) {
            renderMode = GSpecular;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("Depth", &renderType, 6)) {
            renderMode = GDepth;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("Mix", &renderType, 7)) {
            renderMode = GBufferTextureCount;
          }
        }

        if (ImGui::CollapsingHeader("ShadowMap", ImGuiTreeNodeFlags_DefaultOpen)) {
          if(ImGui::SliderFloat("Bias", &shadowMapBias, 0.f, 1.f)) {
            // void ?
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