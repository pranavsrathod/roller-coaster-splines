#version 150

in vec2 tc; 
out vec4 c;
 
uniform sampler2D textureImage;

void main() 
{  
  c = texture(textureImage, tc); 
}