#version 330

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;

out vec3 vViewSpacePosition;
out vec3 vViewSpaceNormal;
out vec2 vTexCoords;
out vec4 vFragPosLightSpace;

uniform mat4 uModelViewProjMatrix;
uniform mat4 uModelViewMatrix;
uniform mat4 uNormalMatrix;
uniform mat4 uLightSpaceMatrix;
uniform mat4 uModelMatrix;

void main()
{
    vViewSpacePosition = vec3(uModelViewMatrix * vec4(aPosition, 1.0));
	vViewSpaceNormal = normalize(vec3(uNormalMatrix * vec4(aNormal, 0.0)));
	vTexCoords = aTexCoords;

    vFragPosLightSpace = uLightSpaceMatrix * uModelMatrix * vec4(aPosition, 1.0);

    gl_Position =  uModelViewProjMatrix * vec4(aPosition, 1);
}