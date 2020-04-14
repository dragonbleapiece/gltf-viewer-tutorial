#version 430

layout (local_size_x = 1, local_size_y = 1) in;

layout (rgba32f, binding = 0) uniform image2D img_output;

uniform sampler2D uPositionTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uAmbientTexture;
uniform sampler2D uDiffuseTexture;
uniform sampler2D uEmissiveTexture;
uniform sampler2D uSpecularTexture;

// For shadow map

uniform mat4 uDirLightViewProjMatrix;
uniform sampler2D uDirLightShadowMap;
uniform float uDirLightShadowMapBias;

// For Depth Map
uniform bool uDisplayDepth;

// Constants
const float GAMMA = 2.2;
const float INV_GAMMA = 1. / GAMMA;

// linear to sRGB approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec3 LINEARtoSRGB(vec3 color)
{
  return pow(color, vec3(INV_GAMMA));
}

void main() {
  // Aucun tableau de donnée n'étant passé au moment de la création de la texture,
  // c'est le compute shader qui va dessiner à l'intérieur de l'image associé
  // à la texture.

  // gl_LocalInvocationID.xy * gl_WorkGroupID.xy == gl_GlobalInvocationID
  ivec2 coords = ivec2(gl_GlobalInvocationID);
  vec4 pixel;

  if(!uDisplayDepth) {
    vec3 diffuse = texelFetch(uDiffuseTexture, coords, 0).rgb;
    vec3 specular = texelFetch(uSpecularTexture, coords, 0).rgb;
    vec3 emissive = texelFetch(uEmissiveTexture, coords, 0).rgb;
    vec3 position = texelFetch(uPositionTexture, coords, 0).rgb;

    vec4 positionInDirLightScreen = uDirLightViewProjMatrix * vec4(position, 1);
    vec3 positionInDirLightNDC = vec3(positionInDirLightScreen / positionInDirLightScreen.w) * 0.5 + 0.5;
    float depthBlockerInDirSpace = texture(uDirLightShadowMap, positionInDirLightNDC.xy).r;
    float dirLightVisibility = positionInDirLightNDC.z < depthBlockerInDirSpace + uDirLightShadowMapBias ? 1.0 : 0.0;


    pixel = vec4(LINEARtoSRGB((diffuse + specular) * dirLightVisibility + emissive), 1.0);

    //pixel = texture(uNormalTexture, coords);

  } else { // display depth map

    float depth = texelFetch(uDirLightShadowMap, coords, 0).r;
    pixel = vec4(vec3(pow(depth, 10)), 1.0);
  
  }
  
  imageStore(img_output, coords, pixel);
}
