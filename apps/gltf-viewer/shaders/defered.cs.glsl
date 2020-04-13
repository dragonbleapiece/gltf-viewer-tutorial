#version 430

layout (local_size_x = 1, local_size_y = 1) in;

layout (rgba32f, binding = 0) uniform image2D img_output;

uniform sampler2D uPositionTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uAmbientTexture;
uniform sampler2D uDiffuseTexture;
uniform sampler2D uEmissiveTexture;
uniform sampler2D uSpecularTexture;

uniform float uViewWidth;
uniform float uViewHeight;

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
  vec2 texCoords = coords / vec2(uViewWidth, uViewHeight);


  vec3 diffuse = texture(uDiffuseTexture, texCoords).rgb;
  vec3 specular = texture(uSpecularTexture, texCoords).rgb;
  vec3 emissive = texture(uEmissiveTexture, texCoords).rgb;

  vec4 pixel = vec4(LINEARtoSRGB((diffuse + specular) + emissive), 1.0);

  //pixel = texture(uNormalTexture, coords);

  imageStore(img_output, coords, pixel);
}
