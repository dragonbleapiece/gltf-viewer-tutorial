// directionalSM.vs.glsl
#version 330

layout(location = 0) in vec3 aPosition;
uniform mat4 uDirLightViewProjMatrix;
uniform mat4 uModelMatrix;

void main()
{
    gl_Position =  uDirLightViewProjMatrix * uModelMatrix * vec4(aPosition, 1);
}
