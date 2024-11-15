#version 150

in vec3 viewPosition; 
in vec3 viewNormal;

out vec4 c;

uniform vec4 La;
uniform vec4 Ld;
uniform vec4 Ls;
uniform vec4 viewLightDirection;

uniform vec4 ka; 
uniform vec4 kd; 
uniform vec4 ks; 
uniform float alpha;

void main()
{
  vec3 eyedir = normalize(vec3(0, 0, 0) - viewPosition);
  vec3 reflectDir = -reflect(viewLightDirection.xyz, viewNormal); 
  float d = max(dot(viewLightDirection.xyz, viewNormal), 0.0f);
  float s = max(dot(reflectDir, eyedir), 0.0f);
  
  c = ka * La + d * kd * Ld + pow(s, alpha) * ks * Ls;
  // c.x = viewNormal.x;
  // c.y = viewNormal.y;
  // c.z = viewNormal.z;
  // c.a = 1.0f;
  
  // if (c.x == 0 && c.y == 0 && c.z == 0){
  // c.x = 1;
  // }
}

