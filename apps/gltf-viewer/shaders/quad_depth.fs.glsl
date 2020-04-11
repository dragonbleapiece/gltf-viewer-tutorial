#version 330
out vec4 fColor;
  
in vec2 vTexCoords;

uniform sampler2D uDepthMap;

void main()
{             
    float depthValue = texture(depthMap, TexCoords).r;
    fColor = vec4(vec3(depthValue), 1.0);
}  