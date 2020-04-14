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
uniform sampler2DShadow uDirLightShadowMap;
uniform sampler2D uDirLightShadowMapOrig;
uniform float uDirLightShadowMapBias;

uniform int uDirLightShadowMapSampleCount;
uniform float uDirLightShadowMapSpread;

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

vec2 poissonDisk[16] = vec2[](
    vec2( -0.94201624, -0.39906216 ),
    vec2( 0.94558609, -0.76890725 ),
    vec2( -0.094184101, -0.92938870 ),
    vec2( 0.34495938, 0.29387760 ),
    vec2( -0.91588581, 0.45771432 ),
    vec2( -0.81544232, -0.87912464 ),
    vec2( -0.38277543, 0.27676845 ),
    vec2( 0.97484398, 0.75648379 ),
    vec2( 0.44323325, -0.97511554 ),
    vec2( 0.53742981, -0.47373420 ),
    vec2( -0.26496911, -0.41893023 ),
    vec2( 0.79197514, 0.19090188 ),
    vec2( -0.24188840, 0.99706507 ),
    vec2( -0.81409955, 0.91437590 ),
    vec2( 0.19984126, 0.78641367 ),
    vec2( 0.14383161, -0.14100790 )
);

float random(vec4 seed)
{
    float dot_product = dot(seed, vec4(12.9898,78.233,45.164,94.673));
    return fract(sin(dot_product) * 43758.5453);
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

    vec4 positionInDirLightScreen = uDirLightViewProjMatrix * vec4(position, 1); // Compute fragment position in NDC space of light
    vec3 positionInDirLightNDC = vec3(positionInDirLightScreen / positionInDirLightScreen.w) * 0.5 + 0.5; // Homogeneize + put between 0 and 1
    
    float dirLightVisibility = 0.0;
    float dirSampleCountf = float(uDirLightShadowMapSampleCount);
    int step = max(1, 16 / uDirLightShadowMapSampleCount);
    for (int i = 0; i < uDirLightShadowMapSampleCount; ++i)
    {

        // Noisy shadows:
        //int index = int(dirSampleCountf * random(vec4(coords, coords.y, i))) % uDirLightShadowMapSampleCount;

        // Blurred shadows:
        int index = (i + step) % uDirLightShadowMapSampleCount;

        dirLightVisibility += textureProj(uDirLightShadowMap, vec4(positionInDirLightNDC.xy + uDirLightShadowMapSpread * poissonDisk[index], positionInDirLightNDC.z - uDirLightShadowMapBias, 1.0), 0.0);
    }
    dirLightVisibility /= dirSampleCountf;

    pixel = vec4(LINEARtoSRGB((diffuse + specular) * dirLightVisibility + emissive), 1.0);

    //pixel = texture(uNormalTexture, coords);

  } else { // display depth map

    float depth = texelFetch(uDirLightShadowMapOrig, coords, 0).r;
    pixel = vec4(vec3(pow(depth, 10)), 1.0);
  
  }
  
  imageStore(img_output, coords, pixel);
}
